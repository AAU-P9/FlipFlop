#!/usr/bin/env python3
import sys
import time
import numpy as np
import json
import os
import abc
import matplotlib.pyplot as plt

_PYCUDA_IMPORT_ERROR = None
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except ImportError as exc:
    _PYCUDA_IMPORT_ERROR = exc
    cuda = None


from gpu_common import GPUArchitecture, NVML_ENABLED, nvmlDeviceGetPowerUsage

try:
    import cupy as cp
    HPC_ENABLED = True
except ImportError:
    HPC_ENABLED = False


class PowerSampler(abc.ABC):
    """Abstract power measurement source."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this sampler can produce readings."""

    @abc.abstractmethod
    def sample_watts(self) -> float:
        """Return the current GPU power draw in watts."""


class NvmlPowerSampler(PowerSampler):
    """Power sampler backed by NVML (pynvml)."""

    def __init__(self, nvml_handle):
        self._handle = nvml_handle

    def is_available(self) -> bool:
        return NVML_ENABLED and self._handle is not None

    def sample_watts(self) -> float:
        return nvmlDeviceGetPowerUsage(self._handle) / 1000.0


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
                 calibration_file="calibration.json",
                 power_sampler: PowerSampler = None):
        self.arch = GPUArchitecture(device_id, calibration_file)
        self.device_name = self.arch.name
        self.arch_key = self.arch.arch_key
        self.runs = runs
        self.idle_sleep = idle_sleep
        self.calibration_file = calibration_file
        if power_sampler is None:
            power_sampler = NvmlPowerSampler(self.arch.nvml_handle)
        self.power_sampler = power_sampler

    # def _plot_power_law(self, xs, ys, alpha, beta):
    #     """Generate and save a plot of SM concurrency power law validation."""
        
    #     if len(xs) == 0 or len(ys) == 0:
    #         print("[WARNING] No data to plot for power law validation.")
    #         return

    #     plt.figure(figsize=(10, 6))
    #     # Plot measured data points
    #     plt.scatter(xs, ys, color='red', label='Measured Power Delta')
    #     # Generate fitted curve
    #     x_fit = np.linspace(min(xs), max(xs), 100)
    #     y_fit = alpha * np.power(x_fit, beta)
    #     plt.plot(x_fit, y_fit, 'b-', label=f'Fitted Curve: {alpha:.2f} * x^{beta:.2f}')
        
    #     plt.xlabel('Number of Active SMs')
    #     plt.ylabel('Power Delta (W)')
    #     plt.title(f'Power Law Validation - {self.device_name}')
    #     plt.legend()
    #     plt.grid(True)

    #     # Save the plot next to the calibration file
    #     base_name = os.path.splitext(self.calibration_file)[0]
    #     plot_path = f"{base_name}_power_law.png"
    #     plt.savefig(plot_path, bbox_inches='tight')
    #     plt.close()
    #     print(f"[INFO] Saved power law validation plot to {plot_path}")
        

    def run_extended_calibration(self):
        print(f"[INFO] Running extended calibration for {self.device_name}")

        # Existing latency/throughput measurements
        overhead_ns   = self._repeat_and_average(self._measure_kernel_launch_overhead)
        lat_coal_ns   = self._repeat_and_average(lambda: self._measure_global_latency(uncoalesced=False))
        lat_uncoal_ns = self._repeat_and_average(lambda: self._measure_global_latency(uncoalesced=True))
        partial_meas = []
        for off in [64, 128, 256, 512]:
            val = self._repeat_and_average(lambda: self._measure_partial_coalescing_latency(offset=off))
            partial_meas.append((off, val))
        partial_slope, partial_intcp = self._fit_linear_regression(partial_meas)
        lat_shared_ns  = self._repeat_and_average(self._measure_shared_latency)
        lat_local_ns   = self._repeat_and_average(self._measure_local_latency)
        issue_cycles   = self._repeat_and_average(self._measure_issue_cycles)
        eff_bw_gbps    = self._repeat_and_average(self._measure_streaming_bandwidth)
        dep_del_coal_s    = (lat_coal_ns   * 1e-9) / 16.0
        dep_del_uncoal_s  = (lat_uncoal_ns * 1e-9) /  8.0

        # Power measurements
        idle_pw = self._repeat_and_average(self._measure_idle_power)
        mem_pw = self._repeat_and_average(self._measure_pure_memory_power)
        fp_pw = self._repeat_and_average(self._measure_pure_fp_power)

        print(f"[DEBUG] idle_pw={idle_pw}, mem_pw={mem_pw}, fp_pw={fp_pw}")

        int_pw = self._repeat_and_average(self._measure_pure_int_power)
        sfu_pw = self._repeat_and_average(self._measure_pure_sfu_power)
        alu_pw = self._repeat_and_average(self._measure_pure_alu_power)
        fds_pw = self._repeat_and_average(self._measure_pure_fds_power)
        reg_pw = self._repeat_and_average(self._measure_pure_reg_power)
        shm_pw = self._repeat_and_average(self._measure_pure_shm_power)

        # Dynamic power components
        max_power_mem = max(0.0, mem_pw - idle_pw)
        max_power_fp = max(0.0, fp_pw - idle_pw)
        max_power_int = max(0.0, int_pw - idle_pw)
        max_power_sfu = max(0.0, sfu_pw - idle_pw)
        max_power_alu = max(0.0, alu_pw - idle_pw)
        max_power_fds = max(0.0, fds_pw - idle_pw)
        max_power_reg = max(0.0, reg_pw - idle_pw)
        max_power_shm = max(0.0, shm_pw - idle_pw)

        print(f"[DEBUG] max_power_mem={max_power_mem}, max_power_fp={max_power_fp}")

        occupancy_shape_param = self._measure_shape_occupancy_factor(idle_pw)


        # Concurrency model
        alpha_fitted, beta_fitted, sms_vals, power_deltas = self._fit_log_scaling_parameters(idle_pw)
        # short_kernel_scale = self._measure_short_kernel_ramp()
        
        # if sms_vals.size > 0 and power_deltas.size > 0:
        #     self._plot_power_law(sms_vals, power_deltas, alpha_fitted, beta_fitted)
        short_kernel_scale = self._measure_short_kernel_ramp()

        new_info = {
            self.arch_key: {
                "baseline_kernel_overhead_ns": overhead_ns,
                "Mem_LD_coal_ns": lat_coal_ns,
                "Mem_LD_uncoal_ns": lat_uncoal_ns,
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
                "max_power_mem": max_power_mem,
                "max_power_fp": max_power_fp,
                "max_power_int": max_power_int,
                "max_power_sfu": max_power_sfu,
                "max_power_alu": max_power_alu,
                "max_power_fds": max_power_fds,
                "max_power_reg": max_power_reg,
                "max_power_shm": max_power_shm,
                "const_sm_power": 0.25,
                "power_alpha": alpha_fitted,
                "power_beta": beta_fitted,
                "short_kernel_scale": short_kernel_scale,
                "max_power_total": 200.0,
                "shape_occupancy_factor": occupancy_shape_param,
                "sm_values": sms_vals.tolist(),
                "power_deltas": power_deltas.tolist(),
            }
        }

        # Save results
        full_data = {}
        if os.path.isfile(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    full_data = json.load(f)
            except:
                print("[WARNING] Could not parse existing calibration file. Overwriting.")
        full_data.update(new_info)
        with open(self.calibration_file, 'w') as ff:
            json.dump(full_data, ff, indent=2)
        print(f"[INFO] Calibration complete for {self.arch_key}")

    # New Microbenchmarks #####################################################
    def _measure_pure_fds_power(self):
        """Measure Fused Datapath Store power"""
        from pycuda.compiler import SourceModule
        mod_src = '''
        __global__ void fds_pure(float *A, int N) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            float val = 1.0f;
            #pragma unroll 16
            for(int i=0; i<1024; i++) {
                val = __fmaf_rn(val, 1.0001f, 0.9999f);
            }
            if(tid < N) A[tid] = val;
        }'''
        return self._run_power_bench(mod_src, "fds_pure")

    def _measure_pure_reg_power(self):
        """Register-intensive operations"""
        from pycuda.compiler import SourceModule
        mod_src = '''
        __global__ void reg_pure(int *A, int N) {
            int r0 = threadIdx.x, r1 = r0+1, r2 = r0+2, r3 = r0+3;
            #pragma unroll 16
            for(int i=0; i<1024; i++) {
                r0 = r0 ^ r1;
                r1 = r1 + r2;
                r2 = r2 * r3;
                r3 = r3 - r0;
            }
            if(blockIdx.x*blockDim.x + threadIdx.x < N) {
                A[threadIdx.x] = r0 + r1 + r2 + r3;
            }
        }'''
        return self._run_power_bench(mod_src, "reg_pure")

    def _measure_pure_shm_power(self):
        """Shared memory intensive kernel"""
        from pycuda.compiler import SourceModule
        mod_src = '''
        __global__ void shm_pure(float *A, int N) {
            __shared__ float smem[1024];
            int tid = threadIdx.x;
            smem[tid] = tid * 0.1f;
            for(int i=0; i<1024; i++) {
                smem[(tid + i) % 1024] = smem[tid] * 1.1f + 
                    smem[(tid + 1) % 1024];
            }
            if(tid < N) A[tid] = smem[tid];
        }'''
        return self._run_power_bench(mod_src, "shm_pure")

    # Utility Methods #########################################################
    def _run_power_bench(self, kernel_src, kernel_name):
        """Generic benchmark runner for power measurement"""
        if not self.power_sampler.is_available():
            return 0.0

        from pycuda.compiler import SourceModule
        mod = SourceModule(kernel_src)
        kernel = mod.get_function(kernel_name)
        N = 32_000_000
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        dA = cuda.mem_alloc(N * 4)

        # Warmup
        for _ in range(3):
            kernel(dA, np.int32(N), block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()

        try:
            samples = []
            start_evt = cuda.Event()
            end_evt = cuda.Event()
            start_evt.record()
            kernel(dA, np.int32(N), block=(block_size,1,1), grid=(grid_size,1,1))

            while not end_evt.query():
                samples.append(self.power_sampler.sample_watts())
                time.sleep(0.01)
                end_evt.record()
                end_evt.synchronize()

            return np.mean(samples) if samples else 0.0
        finally:
            dA.free()

    # =========================================================================
    #  microbenchmark-based measurement for concurrency log scaling
    # =========================================================================
    def _fit_log_scaling_parameters(self, idle_pw):
        """
        Measure average power (above idle) for 1 SM, 2 SM, 4 SM, 8 SM, ...
        Then fit: power = log10(alpha*SM + beta).
        Return (alpha, beta).
        """
        if not self.power_sampler.is_available():
            return (0.1, 1.1)

        sm_list = []
        candidate = 1
        while candidate <= self.arch.sm_count:
            sm_list.append(candidate)
            candidate <<= 1  # powers of two
        if sm_list[-1] < self.arch.sm_count:
            sm_list.append(self.arch.sm_count)

        power_deltas = []
        for sm_count in sm_list:
            pwr = self._measure_power_with_active_sms(sm_count)
            delta = pwr - idle_pw
            power_deltas.append((sm_count, delta))

        import numpy as np
        from scipy.optimize import curve_fit

        # def log_func(sms, alpha, beta):
        #     return np.log10(alpha*sms + beta)

        def power_func(sms, alpha, beta):
            return alpha * np.power(sms, beta)

        xs = np.array([p[0] for p in power_deltas], dtype=np.float32)
        ys = np.array([p[1] for p in power_deltas], dtype=np.float32)
        if xs.size < 2:
            return (0.1, 1.1, xs, ys)

        popt, pcov = curve_fit(power_func, xs, ys, p0=[1.0, 1.0], bounds=(0, np.inf))
        alpha_fitted, beta_fitted = popt
        return (float(alpha_fitted), float(beta_fitted), xs, ys)

    def _measure_power_with_active_sms(self, sm_count):
        """
        Run a kernel that uses only sm_count SMs.
        Return average power in watts.
        """
        from pycuda.compiler import SourceModule
        import time
        src = r'''
        __global__ void sm_limiter(float *A, int N, int loops) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            float val = tid * 0.1f;
            if (tid < N) {
                // Heavy ALU workload (FMA operations)
                for (int j = 0; j < loops; j++) {
                    volatile float v_val = val;
                    v_val = v_val * 1.0001f + 0.9999f;
                    val = v_val;
                }
                A[tid] = val;
            }
        }
        '''
        block_size = 1024
        grid_size = sm_count
        N = block_size*sm_count
        # Keep kernel long enough for power sampling but within int32 range.
        loops_for_kernel = 10_000_000

        arr = np.random.randn(N).astype(np.float32)
        dA = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(dA, arr)

        cc = self.arch.compute_capability
        arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
        mod = SourceModule(src, options=[arch_opt])
        ker = mod.get_function("sm_limiter")

        # Warmup
        for _ in range(2):
            ker(dA, np.int32(N), np.int32(loops_for_kernel),
                block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()

        # measure
        samples = []
        start_evt, end_evt = cuda.Event(), cuda.Event()

        start_evt.record()
        ker(dA, np.int32(N), np.int32(loops_for_kernel),
            block=(block_size,1,1), grid=(grid_size,1,1))
        end_evt.record()

        while not end_evt.query():
            samples.append(self.power_sampler.sample_watts())
            time.sleep(0.01)

        # Wait for full completion
        end_evt.synchronize()
        cuda.Context.synchronize()

        if samples:
            print(f"SM={sm_count}: Power={np.mean(samples):.1f} W (Min={np.min(samples):.1f}, Max={np.max(samples):.1f})")
            return float(np.mean(samples))
        return 60.0  # fallback

    def _measure_short_kernel_ramp(self):
        """
        For short kernels, we rarely reach full dynamic power.
        Compare a single short kernel vs. a repeated longer run.
        Return ratio in [0..1].
        """
        if not self.power_sampler.is_available():
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
                if self.power_sampler.is_available():
                    samples.append(self.power_sampler.sample_watts())
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
            if self.power_sampler.is_available():
                samples.append(self.power_sampler.sample_watts())
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
        """
        if not self.power_sampler.is_available():
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
            samples.append(self.power_sampler.sample_watts())
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
        if not self.power_sampler.is_available():
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
            samples.append(self.power_sampler.sample_watts())
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

    def _measure_shape_occupancy_factor(self, idle_pw):
        """
        Measure how block shape affects occupancy/power efficiency by testing
        multiple aspect ratios (all with 256 total threads) and measuring the
        effective power overhead vs. a square shape.

        Returns:
            shape_factor (float): slope of the linear fit in ln(aspect_ratio) vs.
                                normalized power, clamped [0.1, 0.5].

        Also saves the plot to 'aspect_ratio_penalty.png' and prints a small table
        of (blockX x blockY, aspect_ratio, raw_power, normalized_power).
        """

        if not self.power_sampler.is_available():
            print("[WARN] Power sampler not available; returning fallback shape factor=0.2")
            return 0.2

        from pycuda.compiler import SourceModule
        import time

        # We define 8 shapes, each having 256 threads total: (bx*by=256).
        shapes = [
            (16,16), (32,8), (64,4), (128,2), (256,1),
            (8,32),  (4,64), (2,128)
        ]

        kernel_src = r'''
        __global__ void shape_test(float *data, int N, int loops) {
            int tidx = threadIdx.x + blockIdx.x * blockDim.x;
            int tidy = threadIdx.y + blockIdx.y * blockDim.y;
            int tid = tidx * blockDim.y + tidy;
            if(tid >= N) return;
            float val = data[tid];
            for(int i = 0; i < loops; i++) {
                val = val * 1.0001f + 0.0001f;
            }
            data[tid] = val;
        }
        ''';

        mod = SourceModule(kernel_src)
        kernel = mod.get_function("shape_test")

        N = 1_048_576  # e.g. 1 million
        loops = 50_000_000  # ensure a few seconds so that power stabilizes
        d_data = cuda.mem_alloc(N*4)
        import numpy as np

        host_data = np.random.randn(N).astype(np.float32)
        cuda.memcpy_htod(d_data, host_data)

        # WARMUP: e.g. do one shape to get the GPU 'warmed up'
        kernel(d_data, np.int32(N), np.int32(loops),
            block=(16,16,1), grid=((N // 256),1,1))
        cuda.Context.synchronize()

        shape_results = []  # we'll store (bx, by, aspect_ratio, raw_power_w) here

        for (bx, by) in shapes:
            grid_x = (N + bx*by - 1) // (bx*by)
            samples = []

            start_evt = cuda.Event()
            end_evt   = cuda.Event()

            start_evt.record()
            kernel(d_data, np.int32(N), np.int32(loops),
                block=(bx,by,1), grid=(grid_x,1,1))
            end_evt.record()

            # poll power while kernel is running
            while not end_evt.query():
                samples.append(self.power_sampler.sample_watts())
                time.sleep(0.02)
            end_evt.synchronize()
            cuda.Context.synchronize()

            if not samples:
                print(f"[WARN] No power samples for shape ({bx}x{by}), skipping.")
                continue

            avg_power_w = float(np.mean(samples))
            # Subtract idle to get 'delta', though we can store raw too
            delta_power = avg_power_w - idle_pw
            aspect_ratio = (bx / by) if (bx >= by) else (by / bx)

            shape_results.append((bx, by, aspect_ratio, delta_power))

        d_data.free()

        if not shape_results:
            raise ValueError("[ERROR] No shape data collected. Possibly all shapes got 0 samples?")

        # Identify the 'square' shape data for normalizing
        # We'll store them in arrays for easier fitting/plotting:
        bxs, bys, aspects, delta_powers = zip(*shape_results)

        # Find index of (16,16) among shape_results if present
        sq_idx = None
        for i, (bx,by,_,_) in enumerate(shape_results):
            if bx == 16 and by == 16:
                sq_idx = i
                break

        if sq_idx is None:
            print("[WARN] No valid square block shape data (16,16) found. Using first as baseline.")
            base_power_delta = delta_powers[0]
        else:
            base_power_delta = delta_powers[sq_idx]

        # If the baseline is zero or negative for some reason, clamp it so we don't divide by 0
        if base_power_delta < 1e-5:
            base_power_delta = max(1e-5, base_power_delta)
            print(f"[WARN] The delta power for square shape is very small. Using 1e-5 as fallback baseline.")

        normalized = [dp / base_power_delta for dp in delta_powers]
        ln_aspects = np.log(aspects)

        # If everything is near 1, or if aspect=1 => ln(1)=0 => cluster, that's okay. 
        # We'll just do a linear fit in ln(aspect) vs normalized.
        coeffs = np.polyfit(ln_aspects, normalized, 1)
        slope   = coeffs[0]
        intercept = coeffs[1]
        shape_factor = abs(slope)

        # Print textual table of data
        print("\n[INFO] Occupancy factor data by shape:")
        print("   Shape (bx x by) | Aspect | RawDeltaPower(W) | NormalizedPower")
        for (bx, by, asp, dp), norm in zip(shape_results, normalized):
            print(f"   {bx} x {by:<3}  | {asp:6.2f} | {dp:8.3f}           | {norm:8.3f}")
        print(f"[INFO] Fitted slope in ln(aspect) vs. normalized power = {slope:.4f}, intercept={intercept:.4f}\n")

        # Now produce a more labeled plot
        import matplotlib.pyplot as plt
        x_fit = np.linspace(min(ln_aspects)-0.5, max(ln_aspects)+0.5, 100)
        y_fit = slope * x_fit + intercept

        plt.figure(figsize=(8,5))
        plt.scatter(ln_aspects, normalized, label="Measured Data")
        # Label each point with the shape (bx x by)
        for i,(bx,by) in enumerate(zip(bxs,bys)):
            plt.text(ln_aspects[i]+0.03, normalized[i], f"{bx}x{by}", fontsize=9)

        plt.plot(x_fit, y_fit, 'r--', label=f"Linear Fit: slope={slope:.2f}")
        plt.xlabel("ln(Aspect Ratio)")
        plt.ylabel("Normalized Delta Power")
        plt.title("Non-Square Block Shape Penalty (relative to 16x16)")
        plt.legend()
        plt.grid(True)
        plt.savefig("aspect_ratio_penalty.png", bbox_inches='tight')
        plt.close()
        print("[INFO] Saved aspect_ratio_penalty.png with shape labels.\n")

        # Finally we clamp shape_factor and return it
        final_factor = float(max(0.1, min(shape_factor, 0.5)))

        print(f"[INFO] shape_factor (clamped) = {final_factor:.3f}\n")

        return final_factor


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

        if self.power_sampler.is_available():
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
        if not self.power_sampler.is_available():
            return 50.0
        samples = []
        t0 = time.time()
        while (time.time() - t0) < sample_time_s:
            samples.append(self.power_sampler.sample_watts())
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
            if self.power_sampler.is_available():
                samples.append(self.power_sampler.sample_watts())
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
            if self.power_sampler.is_available():
                samples.append(self.power_sampler.sample_watts())
            time.sleep(0.05)
            end_evt.record()
            end_evt.synchronize()
        cuda.Context.synchronize()

        return float(np.mean(samples)) if samples else None

    def _measure_pure_memory_power(self):
        """Pure memory access with minimal computation"""
        from pycuda.compiler import SourceModule
        mod_src = '''
            __global__ void mem_pure(float *A, float *B, int N) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            // Strided access with wrap-around
            int stride = 32;
            int idx = (tid * stride) % N;
            if (tid < N) {
                B[idx] = A[idx] + 1.0f;
            }
        }'''
        N = 16_000_000  # 128 MB
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        
        A = np.random.randn(N).astype(np.float32)
        dA = cuda.mem_alloc(A.nbytes)
        dB = cuda.mem_alloc(A.nbytes)
        cuda.memcpy_htod(dA, A)
        
        mod = SourceModule(mod_src)
        kernel = mod.get_function("mem_pure")
        
        # Warmup and measure with error handling
        try:
            for _ in range(3):
                kernel(dA, dB, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1, 1))
            cuda.Context.synchronize()
        except Exception as e:
            print(f"Kernel error during warmup: {e}")
            return 0.0
        
        return self._measure_kernel_power(kernel, dA, dB, np.int32(N), 
                                         block=(block_size,1,1), grid=(grid_size,1,1))

    def _measure_pure_fp_power(self):
        """Floating-point MAD operations with high ILP"""
        from pycuda.compiler import SourceModule
        mod_src = '''
        __global__ void fp_pure(float *A, int N) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        float val = tid * 0.1f;
        #pragma unroll 32
        for(int i=0; i<2048; i++) {
            val = __sinf(val) + __cosf(val) * 1.0001f;
        }
        // Ensure result is written to memory
        if(tid < N) A[tid] = val;
        }'''
        N = 1_048_576
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        
        dA = cuda.mem_alloc(N * 4)
        mod = SourceModule(mod_src)
        kernel = mod.get_function("fp_pure")
        
        # Warmup
        for _ in range(3):
            kernel(dA, np.int32(N), block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()
        
        return self._measure_kernel_power(kernel, dA, np.int32(N),
                                         block=(block_size,1,1), grid=(grid_size,1,1))

    def _measure_pure_int_power(self):
        """Integer operations with no memory dependencies"""
        from pycuda.compiler import SourceModule
        mod_src = '''
        __global__ void int_pure(int *A, int N) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int a = 1;
            int b = 2;
            int c = 3;
            #pragma unroll 16
            for(int i=0; i<1024; i++) {
                a = (a * b) ^ c;
                b = (b + c) * a;
                c = (c - a) ^ b;
            }
            if(tid < N) A[tid] = a + b + c;
        }'''
        N = 1_048_576
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        
        dA = cuda.mem_alloc(N * 4)
        mod = SourceModule(mod_src)
        kernel = mod.get_function("int_pure")
        
        # Warmup
        for _ in range(3):
            kernel(dA, np.int32(N), block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()
        
        return self._measure_kernel_power(kernel, dA, np.int32(N),
                                         block=(block_size,1,1), grid=(grid_size,1,1))

    def _measure_pure_sfu_power(self):
        """Special function units (sin/cos/log)"""
        from pycuda.compiler import SourceModule
        mod_src = '''
        __global__ void sfu_pure(float *A, int N) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            float val = 1.0f;
            #pragma unroll 16
            for(int i=0; i<512; i++) {
                val = __sinf(val) + __cosf(val) + __logf(fabsf(val)+1.0f);
            }
            if(tid < N) A[tid] = val;
        }'''
        N = 1_048_576
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        
        dA = cuda.mem_alloc(N * 4)
        mod = SourceModule(mod_src)
        kernel = mod.get_function("sfu_pure")
        
        # Warmup
        for _ in range(3):
            kernel(dA, np.int32(N), block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()
        
        return self._measure_kernel_power(kernel, dA, np.int32(N),
                                         block=(block_size,1,1), grid=(grid_size,1,1))

    def _measure_pure_alu_power(self):
        """ALU operations with branching"""
        from pycuda.compiler import SourceModule
        mod_src = '''
        __global__ void alu_pure(int *A, int N) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int val = tid;
            #pragma unroll 16
            for(int i=0; i<1024; i++) {
                if(val % 2 == 0) val = val * 3 + 1;
                else val = val / 2;
            }
            if(tid < N) A[tid] = val;
        }'''
        N = 1_048_576
        block_size = 256
        grid_size = (N + block_size - 1) // block_size
        
        dA = cuda.mem_alloc(N * 4)
        mod = SourceModule(mod_src)
        kernel = mod.get_function("alu_pure")
        
        # Warmup
        for _ in range(3):
            kernel(dA, np.int32(N), block=(block_size,1,1), grid=(grid_size,1,1))
        cuda.Context.synchronize()
        
        return self._measure_kernel_power(kernel, dA, np.int32(N),
                                         block=(block_size,1,1), grid=(grid_size,1,1))

    def _measure_kernel_power(self, kernel, *args, **kwargs):
        """Generic kernel power measurement"""
        if not self.power_sampler.is_available():
            return 0.0

        samples = []
        start_evt = cuda.Event()
        end_evt = cuda.Event()

        start_evt.record()
        kernel(*args, **kwargs)

        while not end_evt.query():
            samples.append(self.power_sampler.sample_watts())
            time.sleep(0.01)
            end_evt.record()
            end_evt.synchronize()

        return np.mean(samples) if samples else 0.0

def main():
    if _PYCUDA_IMPORT_ERROR is not None:
        print("[ERROR] Missing required Python package: pycuda")
        print("[HINT] Install dependencies from the project root with: uv pip install -r requirements.txt")
        print("[HINT] Run calibration with project env: uv run python experiments/rq2/calibration.py --output calibration.json")
        sys.exit(1)

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
