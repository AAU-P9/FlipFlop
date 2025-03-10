#!/usr/bin/env python3

import sys
import math
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule, compile
import numpy as np
import re
import json
import os
import time
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from pynvml import *
    NVML_ENABLED = True
except ImportError:
    NVML_ENABLED = False

#############################################################################
# 1) Constants & JSON storage
#############################################################################

CALIBRATION_FILE = "calibration.json"

#############################################################################
# 1) GPUArchitecture
#    - loads device attributes, calibration data (both time & power parameters)
#    - can run microbenchmarks for time-latency, and for power stress tests
#############################################################################

class GPUArchitecture:
    """
    Represents a single GPU device’s attributes plus any calibration data
    used by the time and power models.
    """

    def __init__(self, device_id=0):
        self.device = cuda.Device(device_id)
        self.name = self.device.name()
        self.compute_capability = self.device.compute_capability()  # e.g. (8, 6)
        self.attrs = self._fetch_device_attributes()

        # Attempt load calibration from file:
        self.arch_key = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        self.calibration_data = self._load_calibration(CALIBRATION_FILE)

        # For NVML-based power measurement (optional)
        self.nvml_handle = None
        if NVML_ENABLED:
            nvmlInit()
            self.nvml_handle = nvmlDeviceGetHandleByIndex(device_id)

    def _fetch_device_attributes(self) -> Dict:
        """
        Retrieve essential device attributes from PyCUDA.
        """
        da = cuda.device_attribute
        attr_map = {
            'MULTIPROCESSOR_COUNT':
                self.device.get_attribute(da.MULTIPROCESSOR_COUNT),
            'CLOCK_RATE':
                self.device.get_attribute(da.CLOCK_RATE), # kHz
            'GLOBAL_MEMORY_BUS_WIDTH':
                self.device.get_attribute(da.GLOBAL_MEMORY_BUS_WIDTH), # bits
            'MEMORY_CLOCK_RATE':
                self.device.get_attribute(da.MEMORY_CLOCK_RATE), # kHz
            'MAX_THREADS_PER_MULTIPROCESSOR':
                self.device.get_attribute(da.MAX_THREADS_PER_MULTIPROCESSOR),
            'MAX_REGISTERS_PER_MULTIPROCESSOR':
                self.device.get_attribute(da.MAX_REGISTERS_PER_MULTIPROCESSOR),
            'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR':
                self.device.get_attribute(da.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
            'MAX_THREADS_PER_BLOCK':
                self.device.get_attribute(da.MAX_THREADS_PER_BLOCK),
            'REGISTERS_PER_BLOCK':
                self.device.get_attribute(da.MAX_REGISTERS_PER_BLOCK),
            'WARP_SIZE':
                self.device.get_attribute(da.WARP_SIZE), # typically 32
        }
        return attr_map

    def _load_calibration(self, calibration_file: str) -> Dict:
        if os.path.isfile(calibration_file):
            with open(calibration_file, 'r') as cf:
                data = json.load(cf)
            if self.arch_key in data:
                return data[self.arch_key]
            else:
                print(f"[WARNING] {self.arch_key} not found in {calibration_file}.")
                return {}
        else:
            print(f"[WARNING] No calibration file {calibration_file} found.")
            return {}

    @property
    def sm_count(self) -> int:
        return self.attrs['MULTIPROCESSOR_COUNT']

    @property
    def clock_rate_hz(self) -> float:
        # Convert kHz to Hz
        return self.attrs['CLOCK_RATE'] * 1e3

    @property
    def memory_bandwidth_gbps(self) -> float:
        """
        Return memory bandwidth in GB/s (not bytes/s) for convenience.
        memory_clock_rate * bus_width_bits * 2 (DDR) / 8 bits->bytes
        Then convert to gigabytes.
        """
        mem_clk_hz = self.attrs['MEMORY_CLOCK_RATE'] * 1e3
        bw_Bps = (self.attrs['GLOBAL_MEMORY_BUS_WIDTH'] * mem_clk_hz * 2) / 8.0
        return bw_Bps / 1e9

    def has_calibration(self) -> bool:
        return len(self.calibration_data) >= 1

    # ----------------------------------------------------------------
    # run_calibration_and_update_json
    #   Replaces your concurrency-based load test with pointer-chasing
    #   in a single warp to measure actual DRAM latency.
    # ----------------------------------------------------------------
    def run_calibration_and_update_json(self):
        """
        Run microbenchmarks to measure:
          - Global (coalesced) DRAM latency (single-thread pointer chasing)
          - Local/Uncoalesced DRAM latency
          - Shared memory latency
          - Basic arithmetic issue cycles
          - GPU power calibration (idle and stress measurements)
        Then store them in calibration.json as an average.
        """
        print(f"[INFO] Starting calibration for {self.name} (compute cap {self.compute_capability}) ...")
        arch_key = self.arch_key

        # 0) Measure kernel launch overhead
        overhead_ns = self._measure_kernel_launch_overhead()

        # 1) Measure coalesced global DRAM latency (using pointer chasing)
        lat_coal_ns = self._measure_global_latency(uncoalesced=False)

        # 2) Measure uncoalesced global DRAM latency
        lat_uncoal_ns = self._measure_global_latency(uncoalesced=True)

        # 3) Measure shared memory latency
        lat_shared_ns = self._measure_shared_latency()

        # 4) Measure basic arithmetic issue throughput (cycles per instruction)
        measure_issue_cycles = self._measure_issue_cycles()
        if measure_issue_cycles < 1.0:
            measure_issue_cycles = 4.0

        # Compute a partial (average) latency as midpoint between coalesced and uncoalesced
        lat_partial_ns = 0.5 * (lat_coal_ns + lat_uncoal_ns)

        # Define departure delays as fractions of the measured latencies
        dep_del_coal_s = (lat_coal_ns * 1e-9) / 4.0
        dep_del_uncoal_s = (lat_uncoal_ns * 1e-9) / 2.0

        # ----- POWER CALIBRATION -----
        # If NVML is available, run the stress microbenchmarks to measure real power
        if NVML_ENABLED:
            self.run_power_calibration()  # this method updates self.calibration_data with measured power values
            idle_power = self.calibration_data.get("idle_power", 50.0)
            max_power_fp = self.calibration_data.get("max_power_fp", 20.0)
            max_power_int = self.calibration_data.get("max_power_int", 10.0)
            max_power_sfu = self.calibration_data.get("max_power_sfu", 5.0)
            max_power_reg = self.calibration_data.get("max_power_reg", 5.0)
            max_power_mem = self.calibration_data.get("max_power_mem", 30.0)
            const_sm_power = self.calibration_data.get("const_sm_power", 20.0)
            power_log_alpha = self.calibration_data.get("power_log_alpha", 0.1)
            power_log_beta = self.calibration_data.get("power_log_beta", 1.1)
        else:
            # Fallback values if NVML is not available
            idle_power = 50.0
            max_power_fp = 20.0
            max_power_int = 10.0
            max_power_sfu = 5.0
            max_power_reg = 5.0
            max_power_mem = 30.0
            const_sm_power = 20.0
            power_log_alpha = 0.1
            power_log_beta = 1.1

        # Construct the new calibration info using measured values
        new_info = {
            arch_key: {
                "baseline_kernel_overhead_ns": overhead_ns,
                "Mem_LD_coal_ns": lat_coal_ns,
                "Mem_LD_uncoal_ns": lat_uncoal_ns,
                "Mem_LD_partial_ns": lat_partial_ns,
                "issue_cycles": measure_issue_cycles,
                "Departure_del_coal_s": dep_del_coal_s,
                "Departure_del_uncoal_s": dep_del_uncoal_s,
                "Mem_LD_shared_ns": lat_shared_ns,
                "coalesced_bytes": 128,
                "idle_power": idle_power,
                "max_power_fp": max_power_fp,
                "max_power_int": max_power_int,
                "max_power_sfu": max_power_sfu,
                "max_power_reg": max_power_reg,
                "max_power_mem": max_power_mem,
                "const_sm_power": const_sm_power,
                "power_log_alpha": power_log_alpha,
                "power_log_beta": power_log_beta
            }
        }

        # Write the updated calibration info to file
        full_data = {}
        if os.path.isfile(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, 'r') as cf:
                try:
                    full_data = json.load(cf)
                except:
                    full_data = {}
        full_data.update(new_info)
        with open(CALIBRATION_FILE, 'w') as cf:
            json.dump(full_data, cf, indent=2)
        self.calibration_data = new_info[arch_key]
        print(f"[INFO] Wrote updated calibration to {CALIBRATION_FILE} for {arch_key}.")
        print(f"[INFO] Calibration finished: {self.calibration_data}")

    def run_power_calibration(self):
        """
        Uses NVML and stress kernels to measure device power usage.
        This function measures:
          1) Idle power: using NVML over a 3-second interval when no kernel runs.
          2) Memory stress power: launch a memory-bound kernel repeatedly and measure average power.
          3) FP stress power: launch a floating-point–intensive kernel repeatedly.
        Then it computes power constants (max_power values) as the difference between stressed and idle measurements.
        """
        if not NVML_ENABLED or not self.nvml_handle:
            print("[WARNING] NVML not available; skipping power calibration.")
            return

        print(f"[CAL] Starting power calibration for {self.arch_key} using NVML...")

        # 1) Measure idle power
        idle_samples = []
        idle_duration = 3.0  # seconds
        print("[CAL] Measuring idle power for 3 seconds...")
        start_time = time.time()
        while time.time() - start_time < idle_duration:
            p = nvmlDeviceGetPowerUsage(self.nvml_handle)  # in milliwatts
            idle_samples.append(p)
            time.sleep(0.05)
        idle_power_W = np.mean(idle_samples) / 1000.0
        print(f"[CAL] Measured idle power = {idle_power_W:.2f} W")

        # 2) Measure memory-stress power using a memory-bound kernel
        mem_stress_power = self._measure_stress_power(kernel_type="memory", measure_time_sec=3)
        # 3) Measure FP-stress power using an FP-bound kernel
        fp_stress_power = self._measure_stress_power(kernel_type="fp", measure_time_sec=3)

        # Compute the additional power for memory and FP stresses
        max_power_mem = max(mem_stress_power - idle_power_W, 0.0)
        max_power_fp  = max(fp_stress_power - idle_power_W, 0.0)
        # For demonstration, set the other units as fractions of FP stress power
        max_power_int = max_power_fp * 0.5
        max_power_sfu = max_power_fp * 0.3
        max_power_reg = max_power_fp * 0.4
        const_sm_power = max_power_fp * 0.5

        # Set log scaling parameters (you might want to refine these via experiments)
        power_log_alpha = 0.1
        power_log_beta  = 1.1

        new_data = {
            self.arch_key: {
                "idle_power": idle_power_W,
                "max_power_fp": max_power_fp,
                "max_power_int": max_power_int,
                "max_power_sfu": max_power_sfu,
                "max_power_reg": max_power_reg,
                "max_power_mem": max_power_mem,
                "const_sm_power": const_sm_power,
                "power_log_alpha": power_log_alpha,
                "power_log_beta": power_log_beta
            }
        }

        # Merge new power calibration with existing calibration file data.
        all_data = {}
        if os.path.isfile(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, 'r') as f:
                try:
                    all_data = json.load(f)
                except:
                    all_data = {}
        all_data.update(new_data)
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(all_data, f, indent=2)

        # Update local calibration data.
        self.calibration_data.update(new_data[self.arch_key])
        print(f"[CAL] Updated power calibration in {CALIBRATION_FILE} for {self.arch_key}.")
        print(f"[CAL] Power calibration results: {self.calibration_data}")
        
    def _measure_stress_power(self, kernel_type:str, measure_time_sec:int=3) -> float:
        """
        Launch a stress kernel (memory-bound or FP-bound) repeatedly,
        while measuring GPU power via NVML. Return the average measured power in W.
        """
        if kernel_type == "memory":
            kernel_source = r'''
            __global__ void mem_stress(float *A, float *B, int N)
            {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                for(int i = 0; i < 10000; i++){
                    if(idx < N){
                        float tmp = A[idx];
                        B[idx] = tmp + 1.0f;
                    }
                }
            }
            '''
            kernel_name = "mem_stress"
        else:  
            kernel_source = r'''
            __global__ void fp_stress(float *A, float *B, int N)
            {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                float val = 0.0f;
                for(int i = 0; i < 10000; i++){
                    val = val * 1.0001f + 1.0f;
                }
                if(idx < N){
                    B[idx] = val;
                }
            }
            '''
            kernel_name = "fp_stress"

        arch_opt = f"-arch=sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        mod = SourceModule(kernel_source, options=[arch_opt])
        stress_kernel = mod.get_function(kernel_name)

        N = 1024 * 1024
        host_A = np.random.rand(N).astype(np.float32)
        host_B = np.zeros_like(host_A)
        d_A = cuda.mem_alloc(host_A.nbytes)
        d_B = cuda.mem_alloc(host_B.nbytes)
        cuda.memcpy_htod(d_A, host_A)
        cuda.memcpy_htod(d_B, host_B)

        block = (256, 1, 1)
        grid = ((N + block[0] - 1) // block[0], 1)

        # Warm up the kernel
        for _ in range(5):
            stress_kernel(d_A, d_B, np.int32(N), block=block, grid=grid)
        cuda.Context.synchronize()

        # Measure power with NVML over measure_time_sec seconds
        power_samples = []
        start_time = time.time()
        print(f"[CAL] Stressing kernel '{kernel_type}' for {measure_time_sec} seconds ...")
        while time.time() - start_time < measure_time_sec:
            # Launch the stress kernel
            stress_kernel(d_A, d_B, np.int32(N), block=block, grid=grid)
            cuda.Context.synchronize()
            # Read power (in milliwatts)
            p_mW = nvmlDeviceGetPowerUsage(self.nvml_handle)
            power_samples.append(p_mW)
            time.sleep(0.05)
        avg_power_mW = np.mean(power_samples)
        avg_power_W = avg_power_mW / 1000.0
        print(f"[CAL] Average {kernel_type} stress power = {avg_power_W:.2f} W")
        return avg_power_W


    def _measure_kernel_launch_overhead(self)->float:
        """
        Launch a trivially empty kernel many times, measure median time. 
        Return overhead in ns. 
        """
        kernel_src = r'''
        __global__ void emptyKernel(){}
        '''
        arch_opt = f"-arch=sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        mod = SourceModule(kernel_src, options=[arch_opt])
        func = mod.get_function("emptyKernel")
        
        # warmup
        for _ in range(10):
            func(block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        
        times=[]
        start=cuda.Event(); end=cuda.Event()
        N=1000
        for _ in range(N):
            start.record()
            func(block=(1,1,1), grid=(1,1))
            end.record()
            end.synchronize()
            times.append(start.time_till(end)*1e3) # microseconds

        median_us = float(np.median(times))
        overhead_ns = median_us*1e3
        return overhead_ns

    # ----------------------------------------------------------------
    # Pointer chasing approach (1 thread) to measure DRAM latency
    # ----------------------------------------------------------------
    def _measure_global_latency(self, uncoalesced:bool=False) -> float:
        """
        Single-block, single-thread pointer-chasing so concurrency can't hide latency.
        Return the average load latency in ns.
        If uncoalesced=True, we artificially offset addresses.
        """

        mod_source = r'''
        #include <stdio.h>
        __global__ void global_latency(float *buf, int N, int chaseIters, int stride, float *d_out)
        {
            // single-thread or single-warp
            if(threadIdx.x!=0 || blockIdx.x!=0) return;

            int pos = 0;  // start
            float accum=0.0f;
            for(int i=0; i<chaseIters; i++){
                // pointer-chase
                pos = __float_as_int(buf[pos]);  // interpret the float bits as next index
                pos = pos % N; // safety
                accum += pos;
            }
            // store result so it isn't optimized out
            d_out[0] = accum;
        }
        '''
        arch_opt = f"-arch=sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        mod = SourceModule(mod_source, options=[arch_opt])
        kernel = mod.get_function("global_latency")

        # Build pointer-chase array
        N=128*1024
        chaseIters=200000
        arr = np.zeros(N, dtype=np.float32)
        # We'll create a random chain so the addresses are not linearly next
        # If uncoalesced => we add some big stride offset
        strideVal = 1 if not uncoalesced else 37
        pos=0
        for i in range(N):
            pos = (pos + strideVal) % N
            arr[i] = float(pos)

        d_arr = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_arr, arr)

        d_out = cuda.mem_alloc(np.float32(0).nbytes)

        # warmup
        kernel(d_arr, np.int32(N), np.int32(chaseIters), np.int32(strideVal), d_out,
               block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()

        start=cuda.Event(); end=cuda.Event()
        start.record()
        kernel(d_arr, np.int32(N), np.int32(chaseIters), np.int32(strideVal), d_out,
               block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)

        # total load operations: chaseIters
        # The average time per operation:
        # We measure the total kernel time in ms / chaseIters
        per_load_ms = ms / chaseIters
        per_load_ns = per_load_ms*1e6
        return float(per_load_ns)

    # ----------------------------------------------------------------
    # measure_shared_latency
    #   We'll do repeated ld.shared in a single thread
    # ----------------------------------------------------------------
    def _measure_shared_latency(self)->float:
        src=r'''
        __global__ void shared_lat(float *out, int loops)
        {
            __shared__ float sdata[128];
            if(threadIdx.x==0){
                float accum=0;
                for(int i=0; i<128; i++){
                    sdata[i]=(float)i;
                }
                for(int i=0; i<loops; i++){
                    int idx = i%128;
                    accum += sdata[idx];
                }
                out[0]=accum;
            }
        }
        '''
        arch_opt = f"-arch=sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        mod=SourceModule(src, options=[arch_opt])
        func=mod.get_function("shared_lat")
        d_out=cuda.mem_alloc(4)
        loops=2000000

        # warmup
        func(d_out, np.int32(loops), block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()
        start=cuda.Event(); end=cuda.Event()
        start.record()
        func(d_out, np.int32(loops), block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms=start.time_till(end)
        # We have loops shared loads in the single thread
        avg_load_ms= ms/loops
        return float(avg_load_ms*1e6)

    # ----------------------------------------------------------------
    # measure_issue_cycles
    #   We'll do a simple single-block approach but enough threads
    # ----------------------------------------------------------------
    def _measure_issue_cycles(self)->float:
        kernel = r'''
        __global__ void issue_bench(float *data, int loops)
        {
            if(blockIdx.x>0 || threadIdx.x>0) return; // single-thread
            float x = data[0];
            for(int i=0; i<loops; i++){
                // dummy arithmetic
                x = x*1.000001f + 1.0f;
            }
            data[0] = x;
        }
        '''
        arch_opt = f"-arch=sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        mod = SourceModule(kernel, options=[arch_opt])
        func = mod.get_function("issue_bench")

        arr = np.array([1.0], dtype=np.float32)
        d_arr = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_arr, arr)

        loops=10000000
        # warmup
        func(d_arr, np.int32(loops), block=(1,1,1), grid=(1,1))
        cuda.Context.synchronize()

        start=cuda.Event(); end=cuda.Event()
        start.record()
        func(d_arr, np.int32(loops), block=(1,1,1), grid=(1,1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)

        total_insts=loops
        total_secs=ms*1e-3
        cycles_est = total_secs*self.clock_rate_hz
        # cycles / instructions => cycles per thread-instruction
        cpti = cycles_est/float(total_insts)
        # For a warp-level viewpoint, many GPUs can dual-issue or have other intricacies.
        # We'll just do a naive guess that a warp might need ~ cpti * 32 / 2 ...
        # We'll keep it simple: let's just store cpti * warp_size as a "warp instruction" cost
        warp_size = self.attrs['WARP_SIZE'] if self.attrs['WARP_SIZE']>0 else 32
        measure_issue_cycles = cpti*warp_size
        if measure_issue_cycles<1.0:
            measure_issue_cycles=4.0
        return measure_issue_cycles


#############################################################################
# 3) PTX Analysis for modern GPUs (including local/shared detection)
#############################################################################

@dataclass
class KernelAnalysis:
    """
    Data about the PTX code:
      - global_{coal,partial,uncoal} instructions
      - local_mem_insts
      - shared_mem_insts
      - total compute instructions
      - synchronization instructions
      - shared_mem usage
      - registers_per_thread
    """
    mem_coal: int
    mem_uncoal: int
    mem_partial: int
    local_insts: int        # -- CHANGE --
    shared_insts: int       # -- CHANGE --
    comp_insts: int
    synch_insts: int
    shared_mem_bytes: int
    registers_per_thread: int
    total_insts: int = 0

class PTXAnalyzer:
    def __init__(self, ptx_code:str, ptxas_log:str ,arch:GPUArchitecture):
        self.ptx_code = ptx_code
        self.arch = arch
        self.warp_size = self.arch.attrs['WARP_SIZE'] if self.arch.attrs['WARP_SIZE']>0 else 32
        self.ptxas_log = ptxas_log

    def analyze(self)->KernelAnalysis:
        c,u,p = self._classify_global_mem_ops()
        l = self._count_local_mem()
        s = self._count_shared_mem()
        comp = self._count_comp_ops()
        sync = self.ptx_code.count('bar.sync')
        shmem = self._shared_mem_usage()
        regs = self._analyze_registers()

        # We'll do naive total = comp + memory + sync
        naive_total_insts = comp+(c+u+p)+ l + s + sync

        # parse ptxas
        ptxas_info = self._parse_ptxas_info()
        # If ptxas_info includes "Used [REGS] registers", override regs:
        # If ptxas_info includes "Some line with total instructions", override naive_total_insts:
        final_insts = naive_total_insts
        if ptxas_info.get('override_insts',0)> naive_total_insts:
            final_insts = ptxas_info['override_insts']
        if ptxas_info.get('override_regs',0)>regs:
            regs=ptxas_info['override_regs']

        return KernelAnalysis(
            mem_coal=c,
            mem_uncoal=u,
            mem_partial=p,
            local_insts=l,
            shared_insts=s,
            comp_insts=comp,
            synch_insts=sync,
            shared_mem_bytes=shmem,
            registers_per_thread=regs,
            total_insts=final_insts
        )

    def _parse_ptxas_info(self)->Dict:
            """
            Inspect the lines for things like:
            ptxas info    : Used 40 registers, 0 bytes smem, 0 bytes lmem, 8 instructions
            We'll store them in a dict we might call {'override_regs':40, 'override_insts':8} 
            if we find them
            """
            d={}
            lines = self.ptxas_log.split('\n')
            reg_pattern = re.compile(r'Used\s+(\d+)\s+registers')
            inst_pattern= re.compile(r'(\d+)\sinstructions')  # some ptxas versions show "instructions"
            for ln in lines:
                ln=ln.strip().lower()
                m1=reg_pattern.search(ln)
                if m1:
                    d['override_regs']=int(m1.group(1))
                m2=inst_pattern.search(ln)
                if m2:
                    d['override_insts']=int(m2.group(1))
            return d
            
    def _classify_global_mem_ops(self)->Tuple[int,int,int]:
        """
        Distinguish memory ops into coalesced, partial, uncoalesced
        for ld.global, st.global only.
        """
        pattern = re.compile(r'(ld\.global|st\.global)\s.*?\[\s*(%\w+)(?:\s*(\+|\-)\s*(\d+))?\]')
        matches = re.findall(pattern, self.ptx_code)
        coal=0; uncoal=0; partial=0
        for op, reg, sign, offset_str in matches:
            offval=0
            if offset_str is not None and offset_str.strip()!='':
                offval = abs(int(offset_str))
            # We'll define:
            #   if offval==0 => coalesced
            #   if 0<offval< (4*self.warp_size) => partial
            #   else => uncoalesced
            if offval==0:
                coal+=1
            elif offval< (4*self.warp_size):
                partial+=1
            else:
                uncoal+=1
        return (coal, uncoal, partial)

    # -- CHANGE -- to detect local memory load/stores
    def _count_local_mem(self)->int:
        """
        Count ld.local, st.local instructions
        """
        pat = re.compile(r'(ld\.local|st\.local)\b')
        return len(re.findall(pat, self.ptx_code))

    # -- CHANGE -- to detect shared memory load/stores
    def _count_shared_mem(self)->int:
        pat = re.compile(r'(ld\.shared|st\.shared)\b')
        return len(re.findall(pat, self.ptx_code))

    def _count_comp_ops(self)->int:
        lines = self.ptx_code.split('\n')
        c=0
        for ln in lines:
            if ('ld.' in ln) or ('st.' in ln) or ('bar.sync' in ln):
                continue
            # typical arithmetic
            if re.search(r'\b(add|sub|mul|mad|fma|div|neg|abs|and|or|xor)\b', ln):
                c+=1
            elif re.search(r'\.(f32|s32|u32|f64|s64|u64)', ln):
                # naive approach
                c+=1
        return c

    def _shared_mem_usage(self)->int:
        """
        Summation of all .shared arrays in ptx
        """
        pat = re.compile(r'\.shared\s+\.(\w+)\s+(\w+)\s*\[(\d+)\]')
        matches = re.findall(pat, self.ptx_code)
        type_sizes = {'u32':4,'s32':4,'b32':4,'f32':4,'u64':8,'s64':8,'f64':8}
        total=0
        for (dtype, var, count_str) in matches:
            n=int(count_str)
            sz = type_sizes.get(dtype,4)
            total += n*sz
        return total

    def _analyze_registers(self)->int:
        """
        If we see .reg .b32 %r<16>, guess 16 registers used. 
        We'll just return the max found. 
        For real usage, parse the ptxas info or cubin for 'Used N registers'.
        """
        pat = re.compile(r'\.reg\s+\.(b32|s32|u32|f32|b64|s64|u64|f64)\s+%\w+<(\d+)>')
        matches = re.findall(pat, self.ptx_code)
        if not matches:
            return 32
        biggest=0
        for tp, num in matches:
            n = int(num)
            if n>biggest:
                biggest=n
        return biggest

#############################################################################
# 4) ExecutionTimeEstimator
#############################################################################

class ExecutionTimeEstimator:
    """
    We'll incorporate the new local/shared instructions. 
    - local loads typically have same latency as global uncoalesced, 
      but you can store a separate calibration if you like.
    - shared loads have short-lat. We'll read from "Mem_LD_shared_ns" in JSON.
    """

    def __init__(self, arch: GPUArchitecture, analysis: KernelAnalysis, 
                 grid:Tuple[int,int], block:Tuple[int,int]):
        self.arch = arch
        self.analysis = analysis
        self.grid = grid
        self.block = block
        self.warp_size = self.arch.attrs['WARP_SIZE'] if arch.attrs['WARP_SIZE']>0 else 32

        # load from calibration:
        cdata = arch.calibration_data
        # Provide fallback if not found:
        self.baseline_ns   = cdata.get("baseline_kernel_overhead_ns", 0.0)
        self.Mem_coal_ns   = cdata.get("Mem_LD_coal_ns",   500.0)
        self.Mem_uncoal_ns = cdata.get("Mem_LD_uncoal_ns", 800.0)
        self.Mem_partial_ns= cdata.get("Mem_LD_partial_ns",650.0)
        self.Dep_coal_s    = cdata.get("Departure_del_coal_s", 50e-9)
        self.Dep_uncoal_s  = cdata.get("Departure_del_uncoal_s",200e-9)
        self.issue_cycles  = cdata.get("issue_cycles",4)
        self.coalesced_bytes = cdata.get("coalesced_bytes",128)
        # -- CHANGE -- shared memory is short-lat
        self.Mem_shared_ns = cdata.get("Mem_LD_shared_ns", 20.0)

        self.clock_rate = self.arch.clock_rate_hz
        self.sm_count   = self.arch.sm_count

    def get_concurrency_info(self) -> Tuple[float, float]:
        """
        Returns (active_warps_per_sm, total_exec_cycles_for_1_sm) as a helper
        for computing power. 
        We'll convert the predicted time (ns) -> cycles for that single SM.
        """
        # Re-run the same occupancy logic:
        warps_per_block = math.ceil((self.block[0]*self.block[1]) / float(self.warp_size))
        blocks_per_sm = self._calc_blocks_per_sm()
        active_warps = warps_per_block * blocks_per_sm

        # The total predicted kernel time in ns:
        total_ns = self.estimate_time_ns()

        # Approx cycles that one SM effectively experiences:
        # (We just convert the total kernel time -> cycles, ignoring partial idle time.)
        sm_cycles = total_ns * (self.clock_rate / 1e9)

        return (float(active_warps), float(sm_cycles))


    def estimate_time_ns(self) -> float:
        """
        Estimate the total kernel runtime in nanoseconds.

        Improvements:
            - Adds a baseline kernel launch overhead (from calibration)
            - Uses an override for total instruction count (if available from ptxas info)
            - Combines latencies for global (coalesced, uncoalesced, partial),
            local, and shared memory separately.
            - For kernels with no detected global memory traffic (e.g., fully register‐ or shared‐based),
            falls back on a compute-only model.
        """
        # Use the total instruction count from PTX analysis override if available.
        total_insts = self.analysis.total_insts
        if total_insts < 1:
            total_insts = 1

        # Pure compute cost (in cycles)
        comp_cycles = total_insts * self.issue_cycles

        # Determine occupancy: how many blocks per SM can run
        warps_per_block = math.ceil((self.block[0] * self.block[1]) / float(self.warp_size))
        blocks_per_sm = self._calc_blocks_per_sm()
        active_warps = warps_per_block * blocks_per_sm
        total_blocks = self.grid[0] * self.grid[1]
        reps = math.ceil(total_blocks / (blocks_per_sm * self.sm_count))

        # Count all memory instructions (global, local, shared)
        mem_insts = (self.analysis.mem_coal +
                        self.analysis.mem_uncoal +
                        self.analysis.mem_partial +
                        self.analysis.local_insts +
                        self.analysis.shared_insts)

        # If no memory instructions are detected, use compute-only path:
        if mem_insts < 1:
            time_cycles = comp_cycles * reps
            kernel_ns = time_cycles / self.clock_rate * 1e9 + self.baseline_ns
            return kernel_ns

        # === For global memory instructions ===
        n_coal = self.analysis.mem_coal
        n_uncoal = self.analysis.mem_uncoal
        n_part = self.analysis.mem_partial
        global_insts = n_coal + n_uncoal + n_part

        # Convert calibrated latencies (in ns) to seconds:
        lat_coal_s   = self.Mem_coal_ns * 1e-9
        lat_uncoal_s = self.Mem_uncoal_ns * 1e-9
        lat_part_s   = self.Mem_partial_ns * 1e-9

        if global_insts > 0:
            w_coal = float(n_coal) / global_insts
            w_un   = float(n_uncoal) / global_insts
            w_part = float(n_part) / global_insts
        else:
            w_coal = w_un = w_part = 0

        global_lat_s = lat_coal_s * w_coal + lat_uncoal_s * w_un + lat_part_s * w_part

        # === For local and shared memory instructions ===
        local_lat_s  = self.Mem_uncoal_ns * 1e-9    # treat local as uncoalesced
        shared_lat_s = self.Mem_shared_ns * 1e-9      # typically very short

        # Weighted fractions for overall memory instructions:
        if mem_insts > 0:
            frac_global = float(global_insts) / mem_insts
            frac_local  = float(self.analysis.local_insts) / mem_insts
            frac_shared = float(self.analysis.shared_insts) / mem_insts
        else:
            frac_global = frac_local = frac_shared = 0

        # Overall weighted memory latency:
        Mem_L = global_lat_s * frac_global + local_lat_s * frac_local + shared_lat_s * frac_shared

        # === Departure delay for global part ===
        dd_coal   = self.Dep_coal_s
        dd_uncoal = self.Dep_uncoal_s
        dd_part   = 0.5 * (dd_coal + dd_uncoal)
        global_dep = dd_coal * w_coal + dd_uncoal * w_un + dd_part * w_part

        # For local and shared, assume minimal departure delay:
        local_dep = dd_uncoal
        shared_dep = 1.0 / self.clock_rate  # about one cycle
        mem_dep = global_dep * frac_global + local_dep * frac_local + shared_dep * frac_shared

        # === Compute Memory Warp Parallelism (MWP) ===
        active_warps, reps = self._compute_active_blocks_and_reps()
        if mem_dep < 1e-15:
            MWP_woBW_full = active_warps
        else:
            MWP_woBW_full = Mem_L / mem_dep
        MWP_woBW = min(MWP_woBW_full, active_warps)

        # Bandwidth-limited MWP: assume each global memory instruction transfers "coalesced_bytes"
        bps = self.arch.memory_bandwidth_gbps() * 1e9
        warp_bw = self.coalesced_bytes / (global_lat_s if global_lat_s > 1e-15 else 1e-9)
        sm_bw = bps / self.sm_count
        MWP_bw = sm_bw / warp_bw if warp_bw > 1e-9 else active_warps

        MWP = min(MWP_woBW, MWP_bw, active_warps)

        # === Compute memory cycles (for global, local, shared separately) ===
        mem_cycles_coal   = lat_coal_s   * self.clock_rate * n_coal
        mem_cycles_uncoal = lat_uncoal_s * self.clock_rate * n_uncoal
        mem_cycles_part   = lat_part_s   * self.clock_rate * n_part
        mem_global_cy = mem_cycles_coal + mem_cycles_uncoal + mem_cycles_part

        mem_local_cy  = local_lat_s * self.clock_rate * self.analysis.local_insts
        mem_shared_cy = shared_lat_s * self.clock_rate * self.analysis.shared_insts
        mem_total_cy  = mem_global_cy + mem_local_cy + mem_shared_cy

        # === Compute Computation Warp Parallelism (CWP) ===
        if comp_cycles > 1:
            CWP_full = (mem_total_cy + comp_cycles) / comp_cycles
        else:
            CWP_full = active_warps
        CWP = min(CWP_full, active_warps)

        N = active_warps
        # Combine based on Hong & Kim logic
        if abs(MWP - N) < 1e-3 and abs(CWP - N) < 1e-3:
            comp_p = comp_cycles / float(mem_insts)
            time_per_rep_cy = (mem_total_cy + comp_cycles) + comp_p * (MWP - 1)
        elif (CWP >= MWP) or (comp_cycles > mem_total_cy):
            comp_p = comp_cycles / float(mem_insts)
            time_per_rep_cy = mem_total_cy * (N / MWP) + comp_p * (MWP - 1)
        else:
            time_per_rep_cy = (Mem_L * self.clock_rate) + comp_cycles * N

        # Synchronization overhead:
        synch_cost_cy = 0.0
        if self.analysis.synch_insts > 0:
            dep_delay_cy = mem_dep * self.clock_rate
            blocks_per_sm = math.floor(self._calc_blocks_per_sm())
            synch_cost_cy = dep_delay_cy * (MWP - 1) * self.analysis.synch_insts * blocks_per_sm * reps

        total_cycles = time_per_rep_cy * reps + synch_cost_cy

        # Finally add the baseline overhead measured for short kernels.
        total_ns = total_cycles / self.clock_rate * 1e9 + self.baseline_ns
        return total_ns


    def _compute_active_blocks_and_reps(self)->Tuple[float,float]:
        blocks_per_sm = self._calc_blocks_per_sm()
        wpb = math.ceil((self.block[0]*self.block[1]) / float(self.warp_size))
        active_warps = blocks_per_sm*wpb
        total_blocks = float(self.grid[0]*self.grid[1])
        reps = math.ceil(total_blocks/(blocks_per_sm*self.arch.sm_count))
        return (active_warps, reps)

    def _calc_blocks_per_sm(self)->int:
        block_threads = self.block[0]*self.block[1]
        max_threads = self.arch.attrs['MAX_THREADS_PER_MULTIPROCESSOR']
        threads_lim = max_threads//block_threads
        if threads_lim<1: threads_lim=1

        smem_needed = self.analysis.shared_mem_bytes
        smem_lim = self.arch.attrs['MAX_SHARED_MEMORY_PER_MULTIPROCESSOR']
        if smem_needed>0:
            sm_lim = smem_lim//smem_needed
        else:
            sm_lim=16_000
        if sm_lim<1: sm_lim=1

        regs_needed = self.analysis.registers_per_thread*block_threads
        regs_lim = self.arch.attrs['MAX_REGISTERS_PER_MULTIPROCESSOR']
        if regs_needed>0:
            r_lim = regs_lim//regs_needed
        else:
            r_lim=16_000
        if r_lim<1: r_lim=1

        blocks_possible = min(threads_lim, sm_lim, r_lim)
        if blocks_possible>32:
            blocks_possible=32
        return blocks_possible

#############################################################################
# 4) PowerEstimator
#############################################################################

class PowerEstimator:
    """
    Summation of dynamic usage across GPU units, plus idle & log-based factor
    (inspired by the "Integrated GPU Power/Performance" approach).
    """
    def __init__(self, arch: GPUArchitecture, analysis: KernelAnalysis):
        self.arch = arch
        self.analysis = analysis
        cdata = arch.calibration_data
        self.idle_power   = cdata.get("idle_power",50.0)
        self.max_power_fp = cdata.get("max_power_fp",20.0)
        self.max_power_int= cdata.get("max_power_int",10.0)
        self.max_power_sfu= cdata.get("max_power_sfu",5.0)
        self.max_power_reg= cdata.get("max_power_reg",5.0)
        self.max_power_mem= cdata.get("max_power_mem",30.0)
        self.const_sm_power= cdata.get("const_sm_power",20.0)
        self.log_alpha    = cdata.get("power_log_alpha",0.1)
        self.log_beta     = cdata.get("power_log_beta",1.1)

        self.clock_rate   = arch.clock_rate_hz

    def estimate_power(self, exec_cycles: float, warps_per_sm: float, active_sms: int) -> float:
        """
        exec_cycles: total cycles on an SM for the kernel
        warps_per_sm: concurrency
        active_sms: how many SMs used (e.g. if partial usage)
        """
        if exec_cycles<1.0: exec_cycles=1.0
        # classify instructions:
        # you can parse ptx in detail for exact FP/INT/SFU, or guess
        fp_instr  = self._count_fp_insts()
        int_instr = self._count_int_insts()
        sfu_instr = self._count_sfu_insts()
        reg_instr = (self.analysis.comp_insts - (fp_instr+int_instr+sfu_instr))
        mem_instr = (self.analysis.mem_coal + self.analysis.mem_uncoal + self.analysis.mem_partial)

        # AccessRate for each unit
        #   = (# instructions_of_type * warps_per_sm) / (exec_cycles/4)
        factor = warps_per_sm / (exec_cycles/4.0)  # = 4*warps_per_sm / exec_cycles
        rp_fp   = fp_instr*factor*self.max_power_fp
        rp_int  = int_instr*factor*self.max_power_int
        rp_sfu  = sfu_instr*factor*self.max_power_sfu
        rp_reg  = reg_instr*factor*self.max_power_reg
        rp_mem  = mem_instr*factor*self.max_power_mem

        base_sm_dynamic = rp_fp+rp_int+rp_sfu+rp_reg+rp_mem + self.const_sm_power
        from math import log10
        # If you want partial SM approach: power_x = base_sm_dynamic * log10(...(active_sms)...)
        # For a typical "full usage" scenario:
        full_power = base_sm_dynamic*log10(self.log_alpha*active_sms+self.log_beta)
        # Then add idle
        total_gpu_power = self.idle_power + full_power
        return total_gpu_power

    def _count_fp_insts(self)->int:
        # naive guess => half of comp_insts
        return int(self.analysis.comp_insts*0.5)

    def _count_int_insts(self)->int:
        # naive guess => quarter
        return int(self.analysis.comp_insts*0.25)

    def _count_sfu_insts(self)->int:
        # naive guess => 10% 
        return int(self.analysis.comp_insts*0.1)

#############################################################################
# 5) Utility compile & benchmark
#############################################################################

def compile_kernel(kernel_path:str, arch:GPUArchitecture):
    """
    Return (module, ptx, kernel_name).
    We'll do a basic parse to find the kernel entry name in PTX.
    """
    with open(kernel_path,'r') as f:
        source = f.read()
    arch_opt = f"-arch=sm_{arch.compute_capability[0]}{arch.compute_capability[1]}"
    ptx_str = compile(source, target="ptx", options=[arch_opt, "--ptxas-options=-v"]).decode()
    mod = SourceModule(source, options=[arch_opt, "--ptxas-options=-v"])

    log_str = getattr(mod, "_compile_log", "")

    match = re.search(r'\.entry\s+(\w+)', ptx_str)
    if not match:
        raise RuntimeError("Could not find .entry <kernel_name> in the PTX")
    kernel_name = match.group(1)

    # optionally store PTX
    with open(kernel_path+".ptx","w") as ff:
        ff.write(ptx_str)
    return mod, ptx_str, log_str ,kernel_name

def benchmark_kernel_with_power(kernel_func, args, grid: Tuple[int,int], block: Tuple[int,int], runs=50, nvml_handle=None) -> Tuple[float, float]:
    """
    Run the kernel repeatedly while sampling NVML power usage.
    Returns a tuple (median_kernel_time_us, average_power_W).
    If nvml_handle is None, then average_power_W is returned as 0.
    """
    start = cuda.Event(); end = cuda.Event()
    # Warmup runs
    for _ in range(10):
        kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
    cuda.Context.synchronize()

    times = []
    power_samples = []
    for _ in range(runs):
        start.record()
        kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
        end.record()
        end.synchronize()
        t_ms = start.time_till(end)
        times.append(t_ms * 1e3)  # convert ms -> microseconds

        if nvml_handle is not None:
            # Immediately sample power after each kernel run (in milliwatts)
            p_mW = nvmlDeviceGetPowerUsage(nvml_handle)
            power_samples.append(p_mW)
    median_time_us = float(np.median(times))
    if power_samples:
        avg_power_W = float(np.mean(power_samples)) / 1000.0
    else:
        avg_power_W = 0.0
    return median_time_us, avg_power_W

def benchmark_kernel(kernel_func, args, grid:Tuple[int,int], block:Tuple[int,int], runs=50):
    """
    Return median kernel time in microseconds 
    """
    start=cuda.Event(); end=cuda.Event()

    # warmup
    for _ in range(10):
        kernel_func(*args, block=(block[0],block[1],1), grid=(grid[0],grid[1],1))

    times=[]
    for _ in range(runs):
        start.record()
        kernel_func(*args, block=(block[0],block[1],1), grid=(grid[0],grid[1],1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)  # ms
        times.append(ms*1e3)  # convert to microseconds
    return float(np.median(times))

#############################################################################
# 6) Main
#############################################################################

def main():
    if len(sys.argv)<2:
        print("Usage:")
        print("  python time_model.py calibrate")
        print("  python time_model.py run <kernel.cu> <grid_x> <block_x> <size> <runs>")
        sys.exit(0)

    mode = sys.argv[1]
    arch = GPUArchitecture()

    if mode=="calibrate":
        arch.run_calibration_and_update_json()
        return

    elif mode=="run":
        if len(sys.argv)!=7:
            print("Usage: python time_model.py run <kernel.cu> <grid_x> <block_x> <data_size> <runs>")
            sys.exit(1)
        kernel_path=sys.argv[2]
        gx=int(sys.argv[3])
        bx=int(sys.argv[4])
        data_size=int(sys.argv[5])
        runs=int(sys.argv[6])

        if not arch.has_calibration():
            print("[WARNING] No calibration data found. Please run `python time_model.py calibrate` first or fallback values used.")
        
        # compile
        mod, ptx, ptxas_log, kname = compile_kernel(kernel_path, arch)
        kernel_func = mod.get_function(kname)

        # create data
        np_a = np.random.randn(data_size).astype(np.float32)
        np_b = np.random.randn(data_size).astype(np.float32)
        d_a = cuda.mem_alloc(np_a.nbytes)
        d_b = cuda.mem_alloc(np_b.nbytes)
        d_c = cuda.mem_alloc(np_a.nbytes)
        cuda.memcpy_htod(d_a, np_a)
        cuda.memcpy_htod(d_b, np_b)

        # parse PTX
        analyzer = PTXAnalyzer(ptx, ptxas_log, arch)
        analysis = analyzer.analyze()

        # estimate
        estimator = ExecutionTimeEstimator(arch, analysis, (gx,1),(bx,1))
        est_ns = estimator.estimate_time_ns()

        # actual
        kernel_args = [d_a, d_b, d_c, np.int32(data_size)]

        # Actual benchmark: measure time and actual power concurrently if NVML is available
        if NVML_ENABLED and arch.nvml_handle is not None:
            actual_time_us, actual_power_W = benchmark_kernel_with_power(
                kernel_func, kernel_args,
                (gx,1), (bx,1), runs=runs, nvml_handle=arch.nvml_handle)
        else:
            actual_time_us = benchmark_kernel(kernel_func, kernel_args,
                                              (gx,1), (bx,1), runs=runs)
            actual_power_W = 0.0

        actual_ns = actual_time_us*1e3

        diff = abs(est_ns - actual_ns)/max(actual_ns,1e-9)*100.0

        warps_per_sm, sm_cycles = estimator.get_concurrency_info()
        exec_cy = est_ns*(arch.clock_rate_hz/1e9)
        power_estimator = PowerEstimator(arch, analysis)
        p_est = power_estimator.estimate_power(exec_cy, warps_per_sm, arch.sm_count)


        print(f"[RESULT] Kernel = {kname}")
        print(f"  PTX Analysis: {analysis}")
        print(f"  Estimated Time (ns)  : {est_ns:0.2f}")
        print(f"  Actual Median (ns)   : {actual_ns:0.2f}")
        print(f"  Diff (%)             : {diff:0.2f}")
        print(f"  Approx Power (W)     : {p_est:0.2f}")
        print(f"  Actual Power (W)     : {actual_power_W:0.2f}")


    else:
        print(f"Unknown mode: {mode}")


if __name__=="__main__":
    main()
