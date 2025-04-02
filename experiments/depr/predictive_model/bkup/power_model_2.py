#!/usr/bin/env python3
import sys
import math
import time  # Use standard time module for sleep
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from typing import Tuple
from gpu_common import (
    GPUArchitecture,
    PTXAnalyzer,
    KernelAnalysis,
    compile_kernel,
    benchmark_kernel
)
from time_model import ExecutionTimeModel

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetPowerUsage
    )
    NVML_ENABLED = True
except ImportError:
    NVML_ENABLED = False

class PowerEstimator:
    """
    Approximate power model with log-based concurrency scaling.
    """
    def __init__(self, arch: GPUArchitecture, analysis: KernelAnalysis, is_heavy_kernel=False):
        self.arch = arch
        self.analysis = analysis
        cdata = arch.calibration_data

        self.idle_power     = cdata.get("idle_power", 50.0)
        self.max_power_fp   = cdata.get("max_power_fp", 20.0)
        self.max_power_mem  = cdata.get("max_power_mem", 30.0)
        self.max_power_int  = cdata.get("max_power_int", 10.0)
        self.max_power_sfu  = cdata.get("max_power_sfu", 5.0)
        self.max_power_reg  = cdata.get("max_power_reg", 5.0)
        self.const_sm_power = cdata.get("const_sm_power", 20.0)

        self.log_alpha = cdata.get("power_log_alpha", 0.1)
        self.log_beta  = cdata.get("power_log_beta", 1.1)

        self.is_heavy_kernel = is_heavy_kernel

        concurrency_data = cdata.get("concurrency_data", {})
        observed_concurrency = []
        if isinstance(concurrency_data, dict):
            for key in concurrency_data:
                entry = concurrency_data[key]
                if isinstance(entry, dict) and "observed_concurrency" in entry:
                    observed_concurrency.append(entry["observed_concurrency"])
        if observed_concurrency:
            self.concurrency_factor = sum(observed_concurrency) / len(observed_concurrency)
        else:
            self.concurrency_factor = 1.0

    def estimate_power(self, exec_cycles: float, warps_per_sm: float, active_sms: int) -> float:
        if exec_cycles < 1.0:
            exec_cycles = 1.0

        total_comp = self.analysis.comp_insts
        fp_instr  = int(total_comp * 0.5)
        int_instr = int(total_comp * 0.25)
        sfu_instr = int(total_comp * 0.1)
        reg_instr = total_comp - (fp_instr + int_instr + sfu_instr)
        mem_instr = self.analysis.mem_coal + self.analysis.mem_uncoal + self.analysis.mem_partial

        # Dynamic scaling: compute occupancy ratio based on current warps_per_sm
        # Assume MAX_WARPS_PER_SM is provided in calibration or device attributes; default to 64.
        max_warps_per_sm = self.arch.attrs.get('MAX_WARPS_PER_SM', 64)
        occupancy_ratio = warps_per_sm / max_warps_per_sm
        power_scale = 1 + (occupancy_ratio ** 2)  # Quadratic scaling factor

        # Combine HPC concurrency factor with log-based scaling (unchanged)
        concurrency_scale = self.concurrency_factor * self._log_scale_factor(active_sms)

        factor = warps_per_sm / max((exec_cycles / 4.0), 1e-9)

        rp_fp  = fp_instr  * factor * self.max_power_fp
        rp_int = int_instr * factor * self.max_power_int
        rp_sfu = sfu_instr * factor * self.max_power_sfu
        rp_reg = reg_instr * factor * self.max_power_reg
        rp_mem = mem_instr * factor * self.max_power_mem

        # Apply dynamic scaling based on occupancy:
        rp_fp  *= power_scale
        rp_int *= power_scale
        rp_sfu *= power_scale
        rp_reg *= power_scale
        rp_mem *= power_scale

        sm_dynamic_power = rp_fp + rp_int + rp_sfu + rp_reg + rp_mem + self.const_sm_power

        total_dyn_power = sm_dynamic_power * concurrency_scale

        if self.is_heavy_kernel:
            total_dyn_power *= 1.2

        return self.idle_power + total_dyn_power


    def _log_scale_factor(self, active_sms: int) -> float:
        from math import log10
        return log10(self.log_alpha * active_sms + self.log_beta)

def improved_benchmark_kernel_with_power(kernel_func,
                                          args,
                                          grid: Tuple[int,int],
                                          block: Tuple[int,int],
                                          runs=50,
                                          nvml_handle=None,
                                          sampling_interval_s=0.02,  # increased sampling interval
                                          trailing_s=0.2) -> Tuple[float,float]:
    """
    Runs the kernel in a loop and samples power using NVML.
    If the kernel is too short, the loop will extend the measurement period.
    """
    # Warm-up
    for _ in range(10):
        kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
    cuda.Context.synchronize()

    if not nvml_handle:
        times = []
        for _ in range(runs):
            start_evt = cuda.Event()
            end_evt   = cuda.Event()
            start_evt.record()
            kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
            end_evt.record()
            end_evt.synchronize()
            ms = start_evt.time_till(end_evt)
            times.append(ms * 1e3)
        return float(np.median(times)), 0.0

    all_power_samples = []
    run_times_ms = []
    for _ in range(runs):
        start_evt = cuda.Event()
        end_evt = cuda.Event()
        
        # Start power sampling before kernel launch
        start_evt.record()
        # Run kernel multiple times to extend duration
        iterations = 10000  # Increased iterations
        for _ in range(iterations):
            kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
        end_evt.record()
        
        # Sample power until the event completes
        while not end_evt.query():
            p_mW = nvmlDeviceGetPowerUsage(nvml_handle)
            all_power_samples.append(p_mW / 1000.0)
            time.sleep(sampling_interval_s)
        
        # Ensure we capture any trailing power usage
        t_end = time.time() + trailing_s
        while time.time() < t_end:
            p_mW = nvmlDeviceGetPowerUsage(nvml_handle)
            all_power_samples.append(p_mW / 1000.0)
            time.sleep(sampling_interval_s)
        
        # Record the kernel execution time
        end_evt.synchronize()
        ms = start_evt.time_till(end_evt)
        run_times_ms.append(ms * 1e3)

    median_us = float(np.median(run_times_ms))
    avg_power_W = float(np.mean(all_power_samples)) if all_power_samples else 0.0
    return median_us, avg_power_W

def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)
    run_parser = subparsers.add_parser('run', help="Run the power model with improved actual power measurement")

    run_parser.add_argument("kernel", help="kernel file path")
    run_parser.add_argument("gdimx", type=int)
    run_parser.add_argument("gdimy", type=int)
    run_parser.add_argument("bdimx", type=int)
    run_parser.add_argument("bdimy", type=int)
    run_parser.add_argument("size", type=int)
    run_parser.add_argument("runs", type=int)

    args = parser.parse_args()

    if args.mode == "run":
        arch = GPUArchitecture()
        if not arch.has_calibration():
            print("[WARNING] No calibration found; using fallback.")

        mod, ptx_str, ptxas_log, kname = compile_kernel(args.kernel, arch)
        kernel_func = mod.get_function(kname)

        dim = int(math.sqrt(args.size))
        M = dim
        N = dim
        K = dim

        np_a = np.random.randn(M*N).astype(np.float32)
        np_b = np.random.randn(N*K).astype(np.float32)
        np_c = np.zeros((M*K), dtype=np.float32)

        d_a = cuda.mem_alloc(np_a.nbytes)
        d_b = cuda.mem_alloc(np_b.nbytes)
        d_c = cuda.mem_alloc(np_c.nbytes)
        cuda.memcpy_htod(d_a, np_a)
        cuda.memcpy_htod(d_b, np_b)

        analyzer = PTXAnalyzer(ptx_str, ptxas_log, arch, args.bdimx, args.bdimy)
        analysis = analyzer.analyze()

        tm = ExecutionTimeModel(arch, analysis, (args.gdimx, args.gdimy), (args.bdimx, args.bdimy))
        warps_per_sm, sm_cycles = tm.get_concurrency_info()
        est_time_ns = tm.estimate_time_ns()

        is_heavy_kernel = (warps_per_sm > 32)
        pe = PowerEstimator(arch, analysis, is_heavy_kernel=is_heavy_kernel)
        predicted_power = pe.estimate_power(sm_cycles, warps_per_sm, arch.sm_count)

        if NVML_ENABLED and arch.nvml_handle:
            actual_time_us, actual_power_W = improved_benchmark_kernel_with_power(
                kernel_func,
                [d_a, d_b, d_c, np.int32(M), np.int32(N), np.int32(K)],
                grid=(args.gdimx, args.gdimy), block=(args.bdimx, args.bdimy),
                runs=args.runs,
                nvml_handle=arch.nvml_handle
            )
        else:
            actual_time_us = benchmark_kernel(
                kernel_func,
                [d_a, d_b, d_c, np.int32(args.size)],
                (args.gdimx, args.gdimy),
                (args.bdimx, args.bdimy),
                runs=args.runs
            )
            actual_power_W = 0.0

        print(f"[RESULT] Kernel={kname}")
        print(f"PTX Analysis={analysis}")
        print(f"warps_per_sm={warps_per_sm}")
        print(f"sm_cycles={sm_cycles} (from concurrency logic)")
        print(f"Estimated Time (ns)={est_time_ns:.2f}")
        print(f"Predicted Power (W)={predicted_power:.2f}")
        print(f"Actual Time (us)={actual_time_us:.2f}")
        print(f"Actual Power (W)={actual_power_W:.2f}")

        if actual_power_W > 0:
            diff_pct = abs(predicted_power - actual_power_W) / actual_power_W * 100.0
            print(f"Power Diff (%)= {diff_pct:.2f}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
