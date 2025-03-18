#!/usr/bin/env python3
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from gpu_common import GPUArchitecture, compile_kernel, benchmark_kernel_with_power, benchmark_kernel
from time_model import HongKimExecutionTimeModel
from power_model import HongKimPowerEstimator
import os
import json
from PTXAnalyzer import PTXAnalyzer

import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

def prepare_kernel_args(config, arch):
    """
    Prepare kernel arguments from the config file.
    Returns a list of kernel arguments (device pointers or scalar values).
    """
    dims = config.get("dimensions", {})
    # Convert dimension values to integers.
    try:
        resolved_dims = {key: int(value) for key, value in dims.items()}
    except Exception as e:
        raise ValueError(f"Error converting dimensions to integers: {e}")
    
    args_list = []
    for arg in config["kernel_args"]:
        arg_type = arg["type"]
        if arg_type == "array":
            # Resolve each element of the shape.
            shape = []
            for dim in arg["shape"]:
                if isinstance(dim, str):
                    if dim in resolved_dims:
                        shape.append(resolved_dims[dim])
                    else:
                        raise ValueError(f"Dimension '{dim}' not found in config dimensions: {dims}")
                else:
                    shape.append(dim)
            if arg["init"] == "random":
                host_array = np.random.randn(*shape).astype(np.float32)
            elif arg["init"] == "zeros":
                host_array = np.zeros(shape, dtype=np.float32)
            else:
                raise ValueError(f"Unknown init type: {arg['init']}")
            dptr = cuda.mem_alloc(host_array.nbytes)
            cuda.memcpy_htod(dptr, host_array)
            args_list.append(dptr)
        elif arg_type == "int":
            val = arg["value"]
            if isinstance(val, str):
                if val in resolved_dims:
                    val = resolved_dims[val]
                else:
                    raise ValueError(f"Dimension '{val}' not provided in config dimensions: {dims}")
            args_list.append(np.int32(val))
        else:
            raise ValueError(f"Unsupported argument type: {arg_type}")
    return args_list

def main():
    """
    Ties the time and power models together:
      1) Compile and analyze a CUDA kernel.
      2) Predict execution time using the Hong–Kim time model.
      3) Compute concurrency parameters.
      4) Predict GPU power.
      5) Run the kernel to measure actual execution time and average power.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="Path to CUDA kernel source")
    parser.add_argument("gdimx", type=int, help="Grid dimension X")
    parser.add_argument("gdimy", type=int, help="Grid dimension Y")
    parser.add_argument("bdimx", type=int, help="Block dimension X")
    parser.add_argument("bdimy", type=int, help="Block dimension Y")
    parser.add_argument("size", type=int, help="Data size (total number of elements)")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--config", type=str, default="", help="Path to kernel configuration JSON")
    args = parser.parse_args()

    arch = GPUArchitecture()
    mod, ptx, ptxas_log, kname = compile_kernel(args.kernel, arch)

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "kernel_name": "matMul",
            "dimensions": {"M": int(math.sqrt(args.size)), "N": int(math.sqrt(args.size)), "K": int(math.sqrt(args.size))},
            "kernel_args": [
                {"name": "A", "type": "array", "dtype": "float32", "shape": ["M", "N"], "init": "random"},
                {"name": "B", "type": "array", "dtype": "float32", "shape": ["N", "K"], "init": "random"},
                {"name": "C", "type": "array", "dtype": "float32", "shape": ["M", "K"], "init": "zeros"},
                {"name": "M", "type": "int", "value": "M"},
                {"name": "N", "type": "int", "value": "N"},
                {"name": "K", "type": "int", "value": "K"}
            ]
        }

    # Prepare kernel arguments based on configuration.
    kernel_args = prepare_kernel_args(config, arch)

    analyzer = PTXAnalyzer(ptx_code=ptx,
                               ptxas_log=ptxas_log,
                               arch=arch,
                               block_x=args.bdimx,
                               block_y=args.bdimy,
                               config=config)
    analysis = analyzer.analyze()

    # Predict execution time using the Hong–Kim time model
    time_model = HongKimExecutionTimeModel(arch, analysis, (args.gdimx, args.gdimy), (args.bdimx, args.bdimy))
    est_time_ns = time_model.estimate_time_ns()
    exec_cycles = (est_time_ns * arch.clock_rate_hz) / 1e9

    # Compute concurrency: warps per block and blocks per SM
    threads_per_block = args.bdimx * args.bdimy
    warp_size = arch.attrs.get('WARP_SIZE', 32)
    warps_per_block = (threads_per_block + warp_size - 1) // warp_size
    blocks_per_sm = time_model._calc_blocks_per_sm(threads_per_block)
    warps_per_sm = float(blocks_per_sm * warps_per_block)
    active_sms = arch.sm_count

    # Predict power using the Hong–Kim power model
    power_estimator = HongKimPowerEstimator(arch, analysis)
    predicted_power = power_estimator.estimate_power(exec_cycles, warps_per_sm, active_sms, arch.clock_rate_hz)

    # Run the kernel and measure actual time and power if NVML is available
    kernel_func = mod.get_function(kname)
    if arch.nvml_handle is not None:
        actual_us, actual_power = benchmark_kernel_with_power(kernel_func,
                                                              kernel_args,
                                                              grid=(args.gdimx, args.gdimy),
                                                              block=(args.bdimx, args.bdimy),
                                                              runs=args.runs,
                                                              nvml_handle=arch.nvml_handle)
    else:
        actual_us = benchmark_kernel(kernel_func,
                                     kernel_args,
                                     grid=(args.gdimx, args.gdimy),
                                     block=(args.bdimx, args.bdimy),
                                     runs=args.runs)
        actual_power = 0.0

    actual_ns = float(actual_us) * 1e3
    diff_pct = abs(est_time_ns - actual_ns) / max(actual_ns, 1.0) * 100.0

    print(f"[RESULT] Kernel={kname}")
    print(f"Analysis = {analysis}")
    print(f"Estimated Time (ns) = {est_time_ns:.2f}")
    print(f"Actual Time   (ns) = {actual_ns:.2f}")
    print(f"-> Diff (%) = {diff_pct:.2f}")
    print(f"WarpsPerSM = {warps_per_sm:.2f}")
    print(f"Predicted Power (W) = {predicted_power:.2f}")
    print(f"Actual Power (W) = {actual_power:.2f}")

    # -- Additional debug prints for static analysis results--
    print(f"MemCoal={analysis.mem_coal}")
    print(f"MemUncoal={analysis.mem_uncoal}")
    print(f"MemPartial={analysis.mem_partial}")
    print(f"LocalInsts={analysis.local_insts}")
    print(f"SharedInsts={analysis.shared_insts}")
    print(f"SynchInsts={analysis.synch_insts}")
    print(f"FpInsts={analysis.fp_insts}")
    print(f"IntInsts={analysis.int_insts}")
    print(f"SfuInsts={analysis.sfu_insts}")
    print(f"AluInsts={analysis.alu_insts}")
    print(f"TotalInsts={analysis.total_insts}")
    print(f"RegsPerThread={analysis.registers_per_thread}")
    print(f"SharedMemBytes={analysis.shared_mem_bytes}")


if __name__ == "__main__":
    main()
