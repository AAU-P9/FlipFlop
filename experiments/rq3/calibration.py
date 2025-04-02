#!/usr/bin/env python3
import sys
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import json
import os

from gpu_common import GPUArchitecture, NVML_ENABLED, nvmlDeviceGetPowerUsage

try:
    import cupy as cp
    HPC_ENABLED = True
except ImportError:
    HPC_ENABLED = False


class Calibrator:
    """
    A script to run an extended calibration for both time and power on a GPU,
    storing results in calibration.json. All times are measured in ns.
    Additionally, we gather concurrency data (1 SM, 2 SM, etc.) to fit
    a 'log-based' alpha/beta for the concurrency model. We also measure micro-
    benchmark data for integer / FP / memory / sfu, etc. to solve for the
    max_power_* parameters properly instead of hardcoding them.
    """
    def __init__(self, device_id=0, runs=3, idle_sleep=2.0,
                 calibration_file="calibration.json"):
        self.arch = GPUArchitecture(device_id, calibration_file)
        self.device_name = self.arch.name
        self.arch_key = self.arch.arch_key
        self.runs = runs
        self.idle_sleep = idle_sleep
        self.calibration_file = calibration_file

    def run_extended_calibration(self):
        print(f"[INFO] Running extended calibration for {self.device_name} (arch_key={self.arch_key})")

        # (1) Kernel overhead
        overhead_ns   = self._repeat_and_average(self._measure_kernel_launch_overhead)

        # (2) Coalesced & uncoalesced latencies
        lat_coal_ns   = self._repeat_and_average(lambda: self._measure_global_latency(uncoalesced=False))
        lat_uncoal_ns = self._repeat_and_average(lambda: self._measure_global_latency(uncoalesced=True))

        # (3) Partial coalescing linear fit
        partial_offsets = [64, 128, 256, 512]
        partial_meas = []
        for off in partial_offsets:
            val = self._repeat_and_average(lambda: self._measure_partial_coalescing_latency(offset=off))
            partial_meas.append((off, val))
        partial_slope, partial_intcp = self._fit_linear_regression(partial_meas)

        # (4) Shared & local latencies, issue cycles
        lat_shared_ns  = self._repeat_and_average(self._measure_shared_latency)
        lat_local_ns   = self._repeat_and_average(self._measure_local_latency)
        issue_cycles   = self._repeat_and_average(self._measure_issue_cycles)

        # (5) Effective streaming BW
        eff_bw_gbps    = self._repeat_and_average(self._measure_streaming_bandwidth)

        # (6) Example shape occupancy factor
        occupancy_shape_param = self._measure_shape_occupancy_factor()

        # (7) Compute concurrency-based “departure delays”
        dep_del_coal_s    = (lat_coal_ns   * 1e-9) / 16.0
        dep_del_uncoal_s  = (lat_uncoal_ns * 1e-9) /  8.0

        # (8) We measure idle, mem-bound, FP-bound power
        idle_pw, mem_pw, fp_pw = self._measure_power_extremes()

        # (9) Fit concurrency log (e.g. alpha, beta) from 1 SM, 2 SM, etc.
        alpha_fitted, beta_fitted = self._fit_log_scaling_parameters()

        # (10) Optional short-kernel ramp factor
        short_kernel_scale = self._measure_short_kernel_ramp()

        # (11) Solve for integer, SFU, etc. by microbench approach
        int_pw = self._get_int_power_microbench()
        sfu_pw = self._get_sfu_power_microbench()

        # Subtract idle from each so we only store the dynamic portion
        max_power_int = max(0.0, int_pw - idle_pw) if int_pw else 2.0
        max_power_sfu = max(0.0, sfu_pw - idle_pw) if sfu_pw else 2.0

        new_info = {
            self.arch_key: {
                "baseline_kernel_overhead_ns": overhead_ns,
                "Mem_LD_coal_ns":   lat_coal_ns,
                "Mem_LD_uncoal_ns": lat_uncoal_ns,
                "Mem_LD_partial_ns": 0.5*(partial_meas[0][1] + partial_meas[-1][1]),
                "partial_coalesce_slope": partial_slope,
                "partial_coalesce_intercept": partial_intcp,

                "Mem_LD_shared_ns": lat_shared_ns,
                "Mem_LD_local_ns":  lat_local_ns,
                "issue_cycles":     issue_cycles,
                "Departure_del_coal_s":    dep_del_coal_s,
                "Departure_del_uncoal_s":  dep_del_uncoal_s,
                "effective_mem_bw_gbps":   eff_bw_gbps,

                # Fitted / not hardcoded
                "idle_power": idle_pw,
                "max_power_mem": max(0.0, mem_pw - idle_pw),
                "max_power_fp":  max(0.0, fp_pw  - idle_pw),
                "max_power_int": max_power_int,
                "max_power_sfu": max_power_sfu,

                # leftover lumps: minimal
                "max_power_alu": 1.0,
                "max_power_fds": 1.0,
                "max_power_reg": 1.0,
                "max_power_shm": 1.0,

                # minimal const_sm_power
                "const_sm_power": 0.25,

                # concurrency log alpha/beta
                "power_log_alpha": alpha_fitted,
                "power_log_beta":  beta_fitted,

                # partial ramp for short kernels
                "short_kernel_scale": short_kernel_scale,

                "max_power_total": 200.0,

                "shape_occupancy_factor": occupancy_shape_param
            }
        }

        # Merge with existing calibration
        full_data = {}
        if os.path.isfile(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    full_data = json.load(f)
            except:
                print("[WARNING] Could not parse existing calibration file. Overwriting.")
                full_data = {}

        full_data.update(new_info)

        with open(self.calibration_file, 'w') as ff:
            json.dump(full_data, ff, indent=2)
        self.arch.calibration_data = new_info[self.arch_key]
        print(f"[INFO] Calibration updated for {self.arch_key}: {self.arch.calibration_data}")

    # =========================================================================
    #  Illustrative microbenchmark-based measurement for concurrency log scaling
    # =========================================================================
    def _fit_log_scaling_parameters(self):
        """
        Measure average power (above idle) for 1 SM, 2 SM, 4 SM, 8 SM, ...
        Then fit: power = log10(alpha*SM + beta).
        Return (alpha, beta).
        """
        if not NVML_ENABLED or self.arch.nvml_handle is None:
            # fallback
            return (0.1, 1.1)

        sms_list = [1, 2, 4, 8]
        power_deltas = []
        for sm_count in sms_list:
            pwr = self._measure_power_with_active_sms(sm_count)
            base_idle = self.arch.calibration_data.get("idle_power", 50.0)
            delta = pwr - base_idle
            power_deltas.append((sm_count, delta))

        import numpy as np
        from scipy.optimize import curve_fit

        def log_func(sms, alpha, beta):
            return np.log10(alpha*sms + beta)

        xs = np.array([p[0] for p in power_deltas], dtype=np.float32)
        ys = np.array([p[1] for p in power_deltas], dtype=np.float32)
        if xs.size < 2:
            return (0.1, 1.1)

        popt, pcov = curve_fit(log_func, xs, ys, p0=[0.1, 1.1])
        alpha_fitted, beta_fitted = popt
        return (float(alpha_fitted), float(beta_fitted))

    def _measure_power_with_active_sms(self, sm_count):
        """
        Run a kernel that uses only sm_count SMs.
        Return average power in watts.
        """
        from pycuda.compiler import SourceModule
        import time
        src = r'''
        __global__ void sm_limiter(float *A, int N, int loops) {
           int tid = blockDim.x * blockIdx.x + threadIdx.x;
           if(tid<N){
               float tmp = A[tid];
               for(int j=0; j<loops; j++){
                   tmp = tmp*1.00001f + 1.0f;
               }
               A[tid] = tmp;
           }
        }
        '''
        block_size = 256
        N = 256*sm_count
        loops_for_kernel = 100000

        arr = np.random.randn(N).astype(np.float32)
        dA = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(dA, arr)

        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        ker = mod.get_function("sm_limiter")

        # Warmup
        ker(dA, np.int32(N), np.int32(loops_for_kernel),
            block=(block_size,1,1), grid=(sm_count,1,1))
        cuda.Context.synchronize()

        # measure
        samples = []
        start_evt, end_evt = cuda.Event(), cuda.Event()
        start_evt.record()
        ker(dA, np.int32(N), np.int32(loops_for_kernel),
            block=(block_size,1,1), grid=(sm_count,1,1))
        while not end_evt.query():
            if NVML_ENABLED and self.arch.nvml_handle:
                p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
                samples.append(p_mW/1000.0)
            time.sleep(0.05)
            end_evt.record()
            end_evt.synchronize()
        cuda.Context.synchronize()

        if samples:
            return float(np.mean(samples))
        return 60.0  # fallback

    def _measure_short_kernel_ramp(self):
        """
        For short kernels, we rarely reach full dynamic power.
        Compare a single short kernel vs. a repeated longer run.
        Return ratio in [0..1].
        """
        if not NVML_ENABLED or (self.arch.nvml_handle is None):
            return 1.0

        short_pw = self._measure_ultra_short_kernel_power()
        repeated_pw = self._measure_repeated_kernel_power()
        if repeated_pw < 1e-3:
            return 1.0
        ratio = short_pw / repeated_pw
        return float(min(max(ratio, 0.1), 1.0))

    def _measure_ultra_short_kernel_power(self):
        from pycuda.compiler import SourceModule
        import time
        src = "__global__ void emptyK() {}"
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        kfunc = mod.get_function("emptyK")

        samples = []
        for _ in range(5):
            start_evt, end_evt = cuda.Event(), cuda.Event()
            start_evt.record()
            kfunc(block=(1,1,1), grid=(1,1))
            while not end_evt.query():
                if NVML_ENABLED and self.arch.nvml_handle:
                    p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
                    samples.append(p_mW/1000.0)
                time.sleep(0.01)
                end_evt.record()
                end_evt.synchronize()
            cuda.Context.synchronize()
        return float(np.mean(samples)) if samples else 50.0

    def _measure_repeated_kernel_power(self):
        from pycuda.compiler import SourceModule
        import time
        src = "__global__ void emptyK2() { for(int i=0;i<50000;i++) { ; } }"
        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        kfunc = mod.get_function("emptyK2")

        samples = []
        start_evt, end_evt = cuda.Event(), cuda.Event()
        start_evt.record()
        for _ in range(10):
            kfunc(block=(1,1,1), grid=(1,1))
        while not end_evt.query():
            if NVML_ENABLED and self.arch.nvml_handle:
                p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
                samples.append(p_mW/1000.0)
            time.sleep(0.02)
            end_evt.record()
            end_evt.synchronize()
        cuda.Context.synchronize()
        return float(np.mean(samples)) if samples else 60.0


    # --------------------------------------------------------------------------
    #  Below: INT and SFU microbench methods now completed
    # --------------------------------------------------------------------------
    def _get_int_power_microbench(self):
        """
        Run an integer-heavy kernel that saturates integer ALUs
        and measure average power. Return measured power in W.
        This is similar to the existing compute/mem approach, but focusing on int ops.
        """
        if not NVML_ENABLED or (self.arch.nvml_handle is None):
            # fallback
            return None

        from pycuda.compiler import SourceModule
        import time

        mod_src = r'''
        __global__ void int_bound(int *A, int *B, int N, int loops)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if(tid < N){
                int aval = A[tid];
                int bval = B[tid];
                int accum = 0;
                // We do multiple large integer ops in the loop
                for(int j=0; j<loops; j++){
                    for(int i=0; i<20000; i++){
                        // Example integer arithmetic
                        accum = (accum + aval) ^ (bval + i);
                        accum += (aval * bval) ^ (i * 13);
                    }
                }
                A[tid] = accum;
            }
        }
        '''
        block_size = 256
        N = 4_194_304  # 4 million
        loops_for_kernel = 4

        A_host = np.random.randint(0, 50000, size=N).astype(np.int32)
        B_host = np.random.randint(0, 50000, size=N).astype(np.int32)
        dA = cuda.mem_alloc(A_host.nbytes)
        dB = cuda.mem_alloc(B_host.nbytes)
        cuda.memcpy_htod(dA, A_host)
        cuda.memcpy_htod(dB, B_host)

        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(mod_src, options=[arch_opt])
        kernel_func = mod.get_function("int_bound")

        grid_size = (N + block_size - 1)//block_size

        # Warmup
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
            p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
            samples.append(p_mW / 1000.0)
            time.sleep(0.05)
            end_evt.record()
            end_evt.synchronize()
        cuda.Context.synchronize()

        if samples:
            return float(np.mean(samples))
        return None


    def _get_sfu_power_microbench(self):
        """
        Run an SFU kernel (using sin/cos/log/exp or similar)
        to measure power from special function units.
        """
        if not NVML_ENABLED or (self.arch.nvml_handle is None):
            return None

        from pycuda.compiler import SourceModule
        import time

        mod_src = r'''
        __global__ void sfu_bound(float *A, int N, int loops)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if(tid < N){
                float val = A[tid];
                float sumval = 0.0f;
                for(int j=0; j<loops; j++){
                    for(int i=0; i<20000; i++){
                        // e.g. do some sin/cos
                        float s = __sinf(val + i);
                        float c = __cosf(val - i);
                        sumval += s*c + __expf(s);
                    }
                }
                A[tid] = sumval;
            }
        }
        '''
        block_size = 256
        N = 2_097_152  # 2 million
        loops_for_kernel = 4

        A_host = np.random.randn(N).astype(np.float32)
        dA = cuda.mem_alloc(A_host.nbytes)
        cuda.memcpy_htod(dA, A_host)

        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(mod_src, options=[arch_opt])
        kernel_func = mod.get_function("sfu_bound")

        grid_size = (N + block_size - 1)//block_size

        # Warmup
        for _ in range(2):
            kernel_func(dA, np.int32(N), np.int32(loops_for_kernel),
                        block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()

        samples = []
        start_evt, end_evt = cuda.Event(), cuda.Event()
        start_evt.record()
        kernel_func(dA, np.int32(N), np.int32(loops_for_kernel),
                    block=(block_size,1,1), grid=(grid_size,1,1))
        while not end_evt.query():
            p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
            samples.append(p_mW/1000.0)
            time.sleep(0.05)
            end_evt.record()
            end_evt.synchronize()
        cuda.Context.synchronize()

        if samples:
            return float(np.mean(samples))
        return None


    # ===============  Utility & existing code below  ===============

    def _fit_linear_regression(self, offset_lat_pairs):
        """
        offset_lat_pairs is a list of (offset, latency_in_ns).
        We'll do a simple y = slope*x + intercept. Return (slope, intercept).
        """
        import numpy as np
        xs = np.array([p[0] for p in offset_lat_pairs], dtype=np.float32)
        ys = np.array([p[1] for p in offset_lat_pairs], dtype=np.float32)
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
        """Measure how block shape affects occupancy and power efficiency by:
        1. Testing multiple aspect ratios with same total threads
        2. Measuring their effective throughput/power
        3. Fitting a penalty factor for non-square blocks
        """
        if not NVML_ENABLED or (self.arch.nvml_handle is None):
            return 0.2  # fallback

        from pycuda.compiler import SourceModule
        import time
        
        # Test configurations: (block_x, block_y) pairs with 256 total threads
        shapes = [(32,8), (64,4), (128,2), (256,1),
                (16,16), (8,32), (4,64), (2,128)]
        
        kernel_template = """
        __global__ void shape_test(float *data, int N, int loops) {
            int tidx = threadIdx.x + blockIdx.x*blockDim.x;
            int tidy = threadIdx.y + blockIdx.y*blockDim.y;
            int tid = tidx * blockDim.y + tidy;
            
            if(tid >= N) return;
            
            float val = data[tid];
            for(int i=0; i<loops; i++) {
                val = val * 1.0001f + 0.0001f;
            }
            data[tid] = val;
        }
        """
        
        mod = SourceModule(kernel_template)
        kernel = mod.get_function("shape_test")
        N = 1048576  # 1M elements
        loops = 5000
        power_samples = []
        aspect_ratios = []
        
        d_data = cuda.mem_alloc(N * 4)
        host_data = np.random.randn(N).astype(np.float32)
        cuda.memcpy_htod(d_data, host_data)
        
        # Warmup
        kernel(d_data, np.int32(N), np.int32(loops),
            block=(32,8,1), grid=(N//(32*8),1,1))
        cuda.Context.synchronize()
        
        for bx, by in shapes:
            grid_x = (N + (bx*by) - 1) // (bx*by)
            
            # Measure power
            samples = []
            start_evt = cuda.Event()
            end_evt = cuda.Event()
            start_evt.record()
            kernel(d_data, np.int32(N), np.int32(loops),
                block=(bx,by,1), grid=(grid_x,1,1))
            
            while not end_evt.query():
                pwr = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)/1000.0
                samples.append(pwr)
                time.sleep(0.01)
                end_evt.record()
                end_evt.synchronize()
            
            avg_pwr = np.mean(samples) - self.arch.calibration_data["idle_power"]
            power_samples.append(avg_pwr)
            aspect_ratio = bx/by if bx >= by else by/bx
            aspect_ratios.append(aspect_ratio)
            time.sleep(0.5)
        
        # Normalize power samples relative to square block
        base_power = power_samples[shapes.index((16,16))]
        normalized = [p/base_power for p in power_samples]
        
        # Fit logarithmic relationship
        log_aspect = np.log(np.array(aspect_ratios))
        coeffs = np.polyfit(log_aspect, normalized, 1)
        
        # Return slope as shape penalty factor
        shape_factor = abs(coeffs[0])
        return float(max(0.1, min(shape_factor, 0.5)))

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
        arr = np.array([1.0], dtype=np.float32)
        d_arr = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_arr, arr)
        loops = 10_000_000

        # Warmup
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

        instructions_per_loop = 2
        cpti = cycles_est / (loops*instructions_per_loop)
        measure_issue_cycles = cpti
        return measure_issue_cycles if measure_issue_cycles>=1.0 else 4.0

    def _measure_streaming_bandwidth(self) -> float:
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
        grid_size = (N + block_size - 1)//block_size

        # warmup
        for _ in range(5):
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

        import statistics
        median_ms = statistics.median(times_ms)
        bytes_copied = N * 4
        eff_bw_Bps = (bytes_copied / (median_ms * 1e-3))
        return eff_bw_Bps / 1e9

    def _measure_power_extremes(self):
        """
        Measure idle, memory-bound, and FP-bound microbench power. Then
        (optionally) we measure integer, sfu in separate calls. 
        """
        idle_pw = None
        mem_pw  = None
        fp_pw   = None

        if NVML_ENABLED and self.arch.nvml_handle:
            idle_pw = self._repeat_and_average(self._measure_idle_power)
            mem_pw  = self._repeat_and_average(self._measure_mem_bound_power)
            fp_pw   = self._repeat_and_average(self._measure_compute_bound_power)

        if idle_pw is None:
            idle_pw = 50.0
        if mem_pw is None:
            mem_pw = 70.0
        if fp_pw is None:
            fp_pw = 80.0
        return (idle_pw, mem_pw, fp_pw)

    def _measure_idle_power(self, sample_time_s=2.0):
        if not NVML_ENABLED or (self.arch.nvml_handle is None):
            return 50.0
        samples = []
        t0 = time.time()
        while (time.time() - t0) < sample_time_s:
            p_mW = nvmlDeviceGetPowerUsage(self.arch.nvml_handle)
            samples.append(p_mW / 1000.0)
            time.sleep(0.1)
        if samples:
            return float(np.mean(samples))
        return 50.0

    def _measure_mem_bound_power(self):
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

        grid_size = (N + block_size - 1)//block_size

        # warmup
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
        grid_size = (N+block_size-1)//block_size

        # warmup
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
                samples.append(p_mW/1000.0)
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
    parser.add_argument("--output", type=str, default="calibration.json", help="Output calibration file.")
    args = parser.parse_args()

    calibrator = Calibrator(device_id=args.device,
                            runs=args.runs,
                            idle_sleep=args.sleep,
                            calibration_file=args.output)
    calibrator.run_extended_calibration()

if __name__ == "__main__":
    main()
