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


try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
    NVML_ENABLED = True
except ImportError:
    NVML_ENABLED = False
    nvmlInit = None
    nvmlDeviceGetHandleByIndex = None
    nvmlDeviceGetPowerUsage = None

class GPUArchitecture:
    def __init__(self, device_id=0, calibration_file=None):
        self.device = cuda.Device(device_id)
        self.name = self.device.name()
        self.compute_capability = self.device.compute_capability()
        self.attrs = self._fetch_device_attributes()
        self.arch_key = f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"
        self.calibration_file = calibration_file
        self.calibration_data = self._load_calibration(calibration_file)

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
    ptx_bytes = compile(source, target="ptx", options=[arch_opt, "--ptxas-options=-v"], no_extern_c=True)
    ptx_str = ptx_bytes.decode()
    mod = SourceModule(source, options=[arch_opt, "--ptxas-options=-v"], no_extern_c=True)
    compile_log = getattr(mod, "_compile_log", "")
    m = re.search(r"\.entry\s+(\w+)", ptx_str)
    if not m:
        raise RuntimeError("No .entry <kernel> found in PTX!")
    kernel_name = m.group(1)
    return mod, ptx_str, compile_log, kernel_name

