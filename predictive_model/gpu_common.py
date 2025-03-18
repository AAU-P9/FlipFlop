#!/usr/bin/env python3
import os
import json
import re
import math
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import compile, SourceModule
from typing import Dict, Tuple
from dataclasses import dataclass

CALIBRATION_FILE = "calibration.json"

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
    NVML_ENABLED = True
except ImportError:
    NVML_ENABLED = False
    nvmlInit = None
    nvmlDeviceGetHandleByIndex = None
    nvmlDeviceGetPowerUsage = None

class GPUArchitecture:
    def __init__(self, device_id=0):
        self.device = cuda.Device(device_id)
        self.name = self.device.name()
        self.compute_capability = self.device.compute_capability()
        self.attrs = self._fetch_device_attributes()
        self.arch_key = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        self.calibration_data = self._load_calibration(CALIBRATION_FILE)

        if NVML_ENABLED:
            try:
                nvmlInit()
                self.nvml_handle = nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                print(f"[WARNING] NVML init failed: {e}")
                self.nvml_handle = None
        else:
            self.nvml_handle = None

    def _fetch_device_attributes(self) -> Dict:
        da = cuda.device_attribute
        return {
            'MULTIPROCESSOR_COUNT': self.device.get_attribute(da.MULTIPROCESSOR_COUNT),
            'CLOCK_RATE': self.device.get_attribute(da.CLOCK_RATE),  # in kHz
            'GLOBAL_MEMORY_BUS_WIDTH': self.device.get_attribute(da.GLOBAL_MEMORY_BUS_WIDTH),
            'MEMORY_CLOCK_RATE': self.device.get_attribute(da.MEMORY_CLOCK_RATE),  # in kHz
            'MAX_THREADS_PER_MULTIPROCESSOR': self.device.get_attribute(da.MAX_THREADS_PER_MULTIPROCESSOR),
            'MAX_REGISTERS_PER_MULTIPROCESSOR': self.device.get_attribute(da.MAX_REGISTERS_PER_MULTIPROCESSOR),
            'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR': self.device.get_attribute(da.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
            'MAX_THREADS_PER_BLOCK': self.device.get_attribute(da.MAX_THREADS_PER_BLOCK),
            'WARP_SIZE': self.device.get_attribute(da.WARP_SIZE),
            'MAX_BLOCKS_PER_MULTIPROCESSOR': 32  # assume 32 if not provided
        }

    @property
    def sm_count(self) -> int:
        return self.attrs['MULTIPROCESSOR_COUNT']

    @property
    def clock_rate_hz(self) -> float:
        return float(self.attrs['CLOCK_RATE']) * 1e3

    def memory_bandwidth_gbps(self) -> float:
        mem_clk_hz = float(self.attrs['MEMORY_CLOCK_RATE']) * 1e3
        bw_Bps = (self.attrs['GLOBAL_MEMORY_BUS_WIDTH'] * mem_clk_hz * 2.0) / 8.0
        return bw_Bps / 1e9

    def has_calibration(self) -> bool:
        return (len(self.calibration_data) > 0)

    def _load_calibration(self, filename: str) -> Dict:
        if not os.path.isfile(filename):
            print(f"[WARNING] No calibration file '{filename}'. Using empty calibration.")
            return {}
        with open(filename, 'r') as f:
            all_data = json.load(f)
        if self.arch_key in all_data:
            return all_data[self.arch_key]
        else:
            print(f"[WARNING] {self.arch_key} not found in {filename}. Using empty calibration.")
            return {}

@dataclass
class KernelAnalysis:
    mem_coal: int
    mem_uncoal: int
    mem_partial: int
    local_insts: int
    shared_insts: int
    synch_insts: int
    fp_insts: int
    int_insts: int
    sfu_insts: int
    alu_insts: int
    total_insts: int
    registers_per_thread: int
    shared_mem_bytes: int
    block_x: int = 0
    block_y: int = 0


def compile_kernel(kernel_path: str, arch: GPUArchitecture):
    with open(kernel_path, 'r') as f:
        source = f.read()
    cc = arch.compute_capability
    arch_opt = f"-arch=sm_{cc[0]}{cc[1]}"
    ptx_bytes = compile(source, target="ptx", options=[arch_opt, "--ptxas-options=-v"])
    ptx_str = ptx_bytes.decode()
    mod = SourceModule(source, options=[arch_opt, "--ptxas-options=-v"])
    compile_log = getattr(mod, "_compile_log", "")
    m = re.search(r"\.entry\s+(\w+)", ptx_str)
    if not m:
        raise RuntimeError("No .entry <kernel> found in PTX!")
    kernel_name = m.group(1)
    return mod, ptx_str, compile_log, kernel_name


def benchmark_kernel(kernel_func, args, grid: Tuple[int, int], block: Tuple[int, int], runs=50) -> float:
    start = cuda.Event()
    end = cuda.Event()
    for _ in range(10):
        kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
    cuda.Context.synchronize()
    times = []
    for _ in range(runs):
        start.record()
        kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
        end.record()
        end.synchronize()
        ms = start.time_till(end)
        times.append(ms * 1e3)  # microseconds
    return float(np.median(times))

def benchmark_kernel_with_power(kernel_func, args, grid: Tuple[int, int], block: Tuple[int, int],
                                runs=50, nvml_handle=None) -> Tuple[float, float]:
    """
    Launch the kernel and record power usage while it executes.
    For each run:
      - Record start event.
      - Launch the kernel.
      - Record end event immediately.
      - While the kernel is still running (end_evt.query() is False),
        sample NVML power usage every ~20ms.
    Returns:
      (median execution time in microseconds, average power in Watts)
    """
    import time
    start_evt = cuda.Event()
    end_evt = cuda.Event()
    # Warm-up runs
    for _ in range(10):
        kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
    cuda.Context.synchronize()

    time_samples = []
    power_samples = []
    for _ in range(runs):
        start_evt.record()
        kernel_func(*args, block=(block[0], block[1], 1), grid=(grid[0], grid[1], 1))
        end_evt.record()
        # Sample power while kernel is running
        local_power_samples = []
        while not end_evt.query():
            if nvml_handle is not None:
                p_mW = nvmlDeviceGetPowerUsage(nvml_handle)
                local_power_samples.append(p_mW / 1000.0)
            time.sleep(0.02)
        end_evt.synchronize()
        ms = start_evt.time_till(end_evt)
        time_samples.append(ms * 1e3)  # microseconds
        if local_power_samples:
            power_samples.extend(local_power_samples)
    median_time = float(np.median(time_samples))
    avg_power = float(np.mean(power_samples)) if power_samples else 0.0
    return median_time, avg_power
