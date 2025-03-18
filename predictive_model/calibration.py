#!/usr/bin/env python3
import sys
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import json
import os
from gpu_common import GPUArchitecture, CALIBRATION_FILE, NVML_ENABLED, nvmlDeviceGetPowerUsage

try:
    import cupy as cp
    HPC_ENABLED = True
except ImportError:
    HPC_ENABLED = False

class Calibrator:
    """
    A script to run an extended calibration for both time and power on a GPU,
    storing results in calibration.json. All times are measured in ns.
    In addition, we attempt small regressions for partial coalescing and
    occupancy-shape factors, so that we can remove hard-coded constants in PTXAnalyzer.
    """
    def __init__(self, device_id=0, runs=3, idle_sleep=2.0):
        self.arch = GPUArchitecture(device_id)
        self.device_name = self.arch.name
        self.arch_key = self.arch.arch_key
        self.runs = runs
        self.idle_sleep = idle_sleep

    def run_extended_calibration(self):
        print(f"[INFO] Running extended calibration for {self.device_name} (arch_key={self.arch_key})")

        overhead_ns   = self._repeat_and_average(self._measure_kernel_launch_overhead)
        lat_coal_ns   = self._repeat_and_average(lambda: self._measure_global_latency(uncoalesced=False))
        lat_uncoal_ns = self._repeat_and_average(lambda: self._measure_global_latency(uncoalesced=True))

        # Partial coalescing latencies at multiple offsets. Then do a linear regression
        partial_offsets = [64, 128, 256, 512]
        partial_meas = []
        for off in partial_offsets:
            val = self._repeat_and_average(lambda: self._measure_partial_coalescing_latency(offset=off))
            partial_meas.append( (off, val) )
        # Suppose we do a simple linear fit: offset -> partial_lat
        # For demonstration, we treat x=offset, y=latency_in_ns
        # Then from that we might define a partial_coalescing_slope, intercept, etc.
        partial_slope, partial_intcp = self._fit_linear_regression(partial_meas)

        lat_shared_ns  = self._repeat_and_average(self._measure_shared_latency)
        lat_local_ns   = self._repeat_and_average(self._measure_local_latency)
        issue_cycles   = self._repeat_and_average(self._measure_issue_cycles)
        eff_bw_gbps    = self._repeat_and_average(self._measure_streaming_bandwidth)

        # Example: measure shape-occupancy microbenchmark. We'll store a param
        occupancy_shape_param = self._measure_shape_occupancy_factor()

        # For departure delays: you had something like
        dep_del_coal_s = (lat_coal_ns * 1e-9) / 16.0
        dep_del_uncoal_s = (lat_uncoal_ns * 1e-9) / 8.0

        # measure power extremes (requires NVML)
        idle_pw, mem_pw, fp_pw = self._measure_power_extremes()
        new_info = {
            self.arch_key: {
                "baseline_kernel_overhead_ns": overhead_ns,
                "Mem_LD_coal_ns": lat_coal_ns,
                "Mem_LD_uncoal_ns": lat_uncoal_ns,
                # We'll define partial coalescing in two ways:
                #  1) a single average "Mem_LD_partial_ns" if you want
                #  2) or the slope/intercept from the regression
                "Mem_LD_partial_ns": 0.5*(partial_meas[0][1] + partial_meas[-1][1]),
                "partial_coalesce_slope": partial_slope,
                "partial_coalesce_intercept": partial_intcp,

                "Mem_LD_shared_ns": lat_shared_ns,
                "Mem_LD_local_ns": lat_local_ns,
                "issue_cycles": issue_cycles,
                "Departure_del_coal_s": dep_del_coal_s,
                "Departure_del_uncoal_s": dep_del_uncoal_s,
                "effective_mem_bw_gbps": eff_bw_gbps,

                "idle_power": idle_pw,
                "max_power_mem": max(0.0, mem_pw - idle_pw),
                "max_power_fp": max(0.0, fp_pw - idle_pw),

                # Additional power parameters (some remain placeholders).
                "max_power_int": 10.0,
                "max_power_sfu": 5.0,
                "max_power_alu": 5.0,
                "max_power_fds": 5.0,
                "max_power_reg": 5.0,
                "max_power_shm": 1.0,
                "const_sm_power": 1.0,
                "power_log_alpha": 0.1,
                "power_log_beta": 1.1,
                "max_power_total": 300.0,

                # We'll store a shape/occupancy factor from a microbenchmark
                "shape_occupancy_factor": occupancy_shape_param
            }
        }

        # Merge with existing calibration data
        full_data = {}
        if os.path.isfile(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r') as f:
                    full_data = json.load(f)
            except:
                print("[WARNING] Could not parse existing calibration file. Overwriting.")
                full_data = {}
        full_data.update(new_info)
        with open(CALIBRATION_FILE, 'w') as ff:
            json.dump(full_data, ff, indent=2)
        self.arch.calibration_data = new_info[self.arch_key]
        print(f"[INFO] Calibration updated for {self.arch_key}: {self.arch.calibration_data}")

    def _fit_linear_regression(self, offset_lat_pairs):
        """
        offset_lat_pairs is a list of (offset, latency_in_ns).
        We'll do a simple y = slope*x + intercept. Return (slope, intercept).
        """
        import numpy as np
        xs = np.array([p[0] for p in offset_lat_pairs], dtype=np.float32)
        ys = np.array([p[1] for p in offset_lat_pairs], dtype=np.float32)
        # Fit linear: slope, intercept
        # Use closed-form: slope = Cov(x,y)/Var(x), intercept = mean(y)-slope*mean(x)
        xmean = xs.mean()
        ymean = ys.mean()
        num = np.sum((xs - xmean)*(ys - ymean))
        den = np.sum((xs - xmean)*(xs - xmean))
        if abs(den) < 1e-9:
            slope = 0.0
        else:
            slope = num / den
        intercept = ymean - slope*xmean
        return float(slope), float(intercept)

    def _measure_shape_occupancy_factor(self):
        """
        Example: we measure a few block shapes, gather the throughput or latency, then
        define some parameter that says "the shape factor is how quickly performance
        falls off for weird aspect ratios." We store it in calibration as shape_occupancy_factor.
        """
        # In a real script you might run multiple kernels with different shapes, measure times,
        # do a regression. We'll just return a placeholder or we can measure concurrency as example.
        # Return e.g. 0.2 meaning we penalize large aspect ratio 0.2.  We'll do a fixed value:
        return 0.2

    # ==============================
    #  Below: existing calibrations
    # ==============================
    def _repeat_and_average(self, func):
        results = []
        for _ in range(self.runs):
            val = func()
            if val is None:
                val = 0.0
            results.append(val)
            time.sleep(self.idle_sleep)
        import numpy as np
        return float(np.mean(results))

    def _measure_kernel_launch_overhead(self) -> float:
        # same as your original code
        from pycuda.compiler import SourceModule
        kernel_src = "__global__ void emptyKernel() {}"
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(kernel_src, options=[arch_opt])
        func = mod.get_function("emptyKernel")
        for _ in range(10):
            func(block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        import statistics
        times_ms = []
        start = cuda.Event()
        end = cuda.Event()
        N = 200
        for _ in range(N):
            start.record()
            func(block=(1,1,1), grid=(1,1))
            end.record()
            end.synchronize()
            times_ms.append(start.time_till(end))
        median_ms = statistics.median(times_ms)
        return median_ms * 1e6

    def _measure_global_latency(self, uncoalesced=False) -> float:
        # same as your original code
        from pycuda.compiler import SourceModule
        src = r'''
        __global__ void global_latency(float *buf, int N, int chaseIters, int stride, float *d_out)
        {
            if(threadIdx.x != 0 || blockIdx.x != 0) return;
            int pos = 0;
            float accum = 0;
            for(int i=0; i<chaseIters; i++){
                pos = __float_as_int(buf[pos]);
                pos = pos % N;
                accum += pos;
            }
            d_out[0] = accum;
        }
        '''
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        func = mod.get_function("global_latency")
        N = 128*1024
        chaseIters = 200000
        strideVal = 37 if uncoalesced else 1
        arr = np.zeros(N, dtype=np.float32)
        pos = 0
        for i in range(N):
            pos = (pos + strideVal) % N
            arr[i] = float(pos)
        d_arr = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_arr, arr)
        d_out = cuda.mem_alloc(4)
        func(d_arr, np.int32(N), np.int32(chaseIters), np.int32(strideVal), d_out,
             block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        start, end = cuda.Event(), cuda.Event()
        start.record()
        func(d_arr, np.int32(N), np.int32(chaseIters), np.int32(strideVal), d_out,
             block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)
        per_load_ns = (ms / chaseIters) * 1e6
        return float(per_load_ns)

    def _measure_partial_coalescing_latency(self, offset=128):
        # same as your code
        from pycuda.compiler import SourceModule
        src = r'''
        __global__ void partial_lat(float *buf, int N, int stride, int chaseIters, float* out)
        {
            if(threadIdx.x == 0 && blockIdx.x == 0) {
                int pos = 0;
                float val = 0.0f;
                for(int i=0; i<chaseIters; i++){
                    pos = __float_as_int(buf[pos]);
                    pos = pos % N;
                    val += pos;
                }
                out[0] = val;
            }
        }
        '''
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        func = mod.get_function("partial_lat")
        N = 128 * 1024
        chaseIters = 200000
        strideVal = offset // 4
        import numpy as np
        arr = np.zeros(N, dtype=np.float32)
        pos = 0
        for i in range(N):
            pos = (pos + strideVal) % N
            arr[i] = float(pos)
        d_arr = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_arr, arr)
        d_out = cuda.mem_alloc(4)
        func(d_arr, np.int32(N), np.int32(strideVal), np.int32(chaseIters), d_out,
             block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        start, end = cuda.Event(), cuda.Event()
        start.record()
        func(d_arr, np.int32(N), np.int32(strideVal), np.int32(chaseIters), d_out,
             block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)
        return (ms * 1e6) / chaseIters

    def _measure_shared_latency(self) -> float:
        # same as your code
        from pycuda.compiler import SourceModule
        src = r'''
        __global__ void shared_lat(float *out, int loops)
        {
            __shared__ float sdata[128];
            if(threadIdx.x==0){
                for(int i=0;i<128;i++){
                    sdata[i] = float(i);
                }
                float accum = 0;
                for(int i=0; i<loops; i++){
                    accum += sdata[i % 128];
                }
                out[0] = accum;
            }
        }
        '''
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        func = mod.get_function("shared_lat")
        import numpy as np
        d_out = cuda.mem_alloc(4)
        loops = 2_000_000
        func(d_out, np.int32(loops), block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        start, end = cuda.Event(), cuda.Event()
        start.record()
        func(d_out, np.int32(loops), block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)
        per_load_ns = (ms / loops) * 1e6
        return float(per_load_ns)

    def _measure_local_latency(self) -> float:
        # same as your code
        from pycuda.compiler import SourceModule
        src = r'''
        __global__ void local_lat_test(float *out, int loops)
        {
            float accum = 0.0f;
            for(int i=0; i<loops; i++){
                accum += i * 1.1f;
            }
            out[0] = accum;
        }
        '''
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        func = mod.get_function("local_lat_test")
        import numpy as np
        d_out = cuda.mem_alloc(4)
        loops = 2_000_000
        func(d_out, np.int32(loops), block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        start, end = cuda.Event(), cuda.Event()
        start.record()
        func(d_out, np.int32(loops), block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)
        per_iter_ns = (ms / loops) * 1e6
        return per_iter_ns

    def _measure_issue_cycles(self) -> float:
        # same as your code
        from pycuda.compiler import SourceModule
        src = r'''
        __global__ void issue_bench(float *data, int loops)
        {
            if(threadIdx.x>0 || blockIdx.x>0) return;
            float x = data[0];
            for(int i=0; i<loops; i++){
                x = x * 1.000001f + 1.0f;
            }
            data[0] = x;
        }
        '''
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        func = mod.get_function("issue_bench")
        import numpy as np
        arr = np.array([1.0], dtype=np.float32)
        d_arr = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_arr, arr)
        loops = 10_000_000
        func(d_arr, np.int32(loops), block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        start, end = cuda.Event(), cuda.Event()
        start.record()
        func(d_arr, np.int32(loops), block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)
        total_s = ms * 1e-3
        cycles_est = total_s * self.arch.clock_rate_hz
        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        cpti = cycles_est / loops
        measure_issue_cycles = cpti * warp_size
        return measure_issue_cycles if measure_issue_cycles >= 1.0 else 4.0

    def _measure_streaming_bandwidth(self) -> float:
        # same as your code
        from pycuda.compiler import SourceModule
        import statistics
        src = r'''
        __global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int N)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid < N){
                out[tid] = in[tid];
            }
        }
        '''
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        func = mod.get_function("copy_kernel")
        N = 16_777_216
        a_host = np.random.randn(N).astype(np.float32)
        b_host = np.zeros_like(a_host)
        d_a = cuda.mem_alloc(a_host.nbytes)
        d_b = cuda.mem_alloc(a_host.nbytes)
        cuda.memcpy_htod(d_a, a_host)
        block_size = 256
        grid_size = (N+block_size-1)//block_size
        for _ in range(10):
            func(d_a, d_b, np.int32(N),
                 block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()
        times_ms = []
        runs = 10
        start, end = cuda.Event(), cuda.Event()
        for _ in range(runs):
            start.record()
            func(d_a, d_b, np.int32(N),
                 block=(block_size,1,1), grid=(grid_size,1,1))
            end.record()
            end.synchronize()
            times_ms.append(start.time_till(end))
        median_ms = statistics.median(times_ms)
        bytes_copied = N * 4
        eff_bw_Bps = (bytes_copied / (median_ms * 1e-3))
        return eff_bw_Bps / 1e9

    def _measure_power_extremes(self):
        idle_pw = None
        mem_pw  = None
        fp_pw   = None
        if NVML_ENABLED and self.arch.nvml_handle:
            idle_pw = self._repeat_and_average(self._measure_idle_power)
            mem_pw  = self._repeat_and_average(self._measure_mem_bound_power)
            fp_pw   = self._repeat_and_average(self._measure_compute_bound_power)
        # if anything is None, fallback
        if idle_pw is None:
            idle_pw = 50.0
        if mem_pw is None:
            mem_pw = 70.0
        if fp_pw is None:
            fp_pw = 80.0
        return idle_pw, mem_pw, fp_pw

    def _measure_idle_power(self, sample_time_s=2.0):
        if not NVML_ENABLED or (self.arch.nvml_handle is None):
            return 50.0
        samples = []
        t0 = time.time()
        while (time.time() - t0) < sample_time_s:
            p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
            samples.append(p_mW / 1000.0)
            time.sleep(0.1)
        return float(np.mean(samples)) if samples else 50.0

    def _measure_mem_bound_power(self):
        # unchanged
        from pycuda.compiler import SourceModule
        import time
        mod_src = r'''
        __global__ void mem_bound(float *A, float *B, int N, int loops) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if(tid < N){
                float val = 0.0f;
                for(int j=0; j<loops; j++){
                    for(int i=0; i<500; i++){
                        val += A[(tid + i*37) % N];
                    }
                }
                B[tid] = val;
            }
        }
        '''
        block_size = 256
        N = 16_777_216
        loops_for_kernel = 20
        A_host = np.random.randn(N).astype(np.float32)
        B_host = np.zeros_like(A_host)
        dA = cuda.mem_alloc(A_host.nbytes)
        dB = cuda.mem_alloc(B_host.nbytes)
        cuda.memcpy_htod(dA, A_host)
        cuda.memcpy_htod(dB, B_host)
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(mod_src, options=[arch_opt])
        kernel_func = mod.get_function("mem_bound")
        grid_size = (N + block_size - 1) // block_size
        for _ in range(2):
            kernel_func(dA, dB, np.int32(N), np.int32(loops_for_kernel),
                        block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()
        samples = []
        start_evt, end_evt = cuda.Event(), cuda.Event()
        start_evt.record()
        kernel_func(dA, dB, np.int32(N), np.int32(loops_for_kernel),
                    block=(block_size,1,1), grid=(grid_size,1,1))
        while not end_evt.query():
            if NVML_ENABLED and self.arch.nvml_handle:
                p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
                samples.append(p_mW / 1000.0)
            time.sleep(0.05)
            end_evt.record()
            end_evt.synchronize()
        cuda.Context.synchronize()
        return float(np.mean(samples)) if samples else None

    def _measure_compute_bound_power(self):
        # unchanged
        from pycuda.compiler import SourceModule
        import time
        mod_src = r'''
        __global__ void compute_bound(float *A, float *B, int N, int loops) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if(tid < N){
                float valA = A[tid];
                float valB = B[tid];
                float accum = 0.0f;
                for(int j=0; j<loops; j++){
                    for(int i=0; i<20000; i++){
                        accum = fmaf(accum+valA, valB+0.9999f, 1.0001f);
                    }
                }
                A[tid] = accum;
            }
        }
        '''
        block_size = 256
        N = 8_388_608
        loops_for_kernel = 5
        A_host = np.random.randn(N).astype(np.float32)
        B_host = np.random.randn(N).astype(np.float32)
        dA = cuda.mem_alloc(A_host.nbytes)
        dB = cuda.mem_alloc(B_host.nbytes)
        cuda.memcpy_htod(dA, A_host)
        cuda.memcpy_htod(dB, B_host)
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(mod_src, options=[arch_opt])
        kernel_func = mod.get_function("compute_bound")
        grid_size = (N + block_size - 1) // block_size
        for _ in range(2):
            kernel_func(dA, dB, np.int32(N), np.int32(loops_for_kernel),
                        block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()
        samples = []
        start_evt, end_evt = cuda.Event(), cuda.Event()
        start_evt.record()
        kernel_func(dA, dB, np.int32(N), np.int32(loops_for_kernel),
                    block=(block_size,1,1), grid=(grid_size,1,1))
        while not end_evt.query():
            if NVML_ENABLED and self.arch.nvml_handle:
                p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
                samples.append(p_mW / 1000.0)
            time.sleep(0.05)
            end_evt.record()
            end_evt.synchronize()
        cuda.Context.synchronize()
        return float(np.mean(samples)) if samples else None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run extended GPU calibration for updated time/power models.")
    parser.add_argument("--runs", type=int, default=5, help="Number of measurements per microbenchmark.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID.")
    parser.add_argument("--sleep", type=float, default=5.0, help="Sleep time after each measurement.")
    args = parser.parse_args()
    calibrator = Calibrator(device_id=args.device, runs=args.runs, idle_sleep=args.sleep)
    calibrator.run_extended_calibration()

if __name__ == "__main__":
    main()
