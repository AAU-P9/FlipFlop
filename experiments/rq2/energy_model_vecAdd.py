#!/usr/bin/env python3
import argparse
import os
import math
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import kernel_tuner as kt
from kernel_tuner.observers.nvml import NVMLObserver

import pycuda.driver as pcudadriver
import pycuda.autoinit

from gpu_common import GPUArchitecture, compile_kernel
from time_model import HongKimExecutionTimeModel
from power_model import HongKimPowerEstimator
from PTXAnalyzer import PTXAnalyzer

def generate_block_combinations():
    """Generate all valid 1D block size combinations for vecAdd"""
    block_sizes = []
    for x in [32, 64, 128, 256, 512, 1024]:
        if x >= 32 and x <= 1024 and x % 32 == 0:
            block_sizes.append((x,))
    return block_sizes

def prepare_kernel_args(n):
    """Generate kernel arguments for vecAdd with vector size n"""
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    
    return [a, b, c, np.int32(n)]

def run_configuration(kernel_path, kernel_src, arch, n, iterations):
    """Run energy model predictions and measurements for all valid block sizes"""
    args = prepare_kernel_args(n)
    block_combos = generate_block_combinations()
    
    results = []
    for (block_x,) in block_combos:
        # Compile and analyze
        mod, ptx_str, ptxas_log, kname = compile_kernel(kernel_path, arch)
        analyzer = PTXAnalyzer(ptx_str, ptxas_log, arch, block_x, 1, {})
        analysis = analyzer.analyze()
        
        # Calculate grid dimensions
        grid_x = (n + block_x - 1) // block_x  # Number of blocks needed to cover n elements
        grid_y = 1
        
        # Time prediction
        time_model = HongKimExecutionTimeModel(
            arch, analysis, (grid_x, grid_y), (block_x, 1))
        est_time_ns = time_model.estimate_time_ns()
        
        # Power prediction
        warp_size = arch.attrs.get('WARP_SIZE', 32)
        warps_per_block = (block_x + warp_size - 1) // warp_size
        blocks_per_sm = time_model._calc_blocks_per_sm(block_x)
        warps_per_sm = blocks_per_sm * warps_per_block
        
        power_estimator = HongKimPowerEstimator(arch, analysis)
        predicted_power = power_estimator.estimate_power(
            (est_time_ns * arch.clock_rate_hz) / 1e9, warps_per_sm, arch.sm_count, arch.clock_rate_hz)
        
        compiler_options = [
        "--std=c++17",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
        ]

        # Actual measurements
        nvml_observer = NVMLObserver(["nvml_energy", "nvml_power"])

        result, _ = kt.tune_kernel(
            kernel_name=kname,
            kernel_source=kernel_src,
            problem_size=(n, 1),
            arguments=args,
            tune_params={"block_size_x": [block_x]},
            compiler_options=compiler_options,
            observers=[nvml_observer],
            metrics={"time_ns": lambda p: p["time"] * 1e6},
            iterations=iterations,
            verbose=False
        )

        # Collect results
        entry = {
            "vector_size": n,
            "block_x": block_x,
            "thread_count": block_x,
            "predicted_time_ns": est_time_ns,
            "predicted_power": predicted_power,
            "actual_time_ns": result[0]["time_ns"],
            "actual_power": np.mean(result[0]["nvml_power"]),
            **analysis.__dict__
        }
        results.append(entry)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="VecAdd Kernel Energy Analysis")
    parser.add_argument("--kernel_file", required=True, help="Path to CUDA kernel")
    parser.add_argument("--vector_sizes", type=str, help="Comma-separated vector sizes")
    parser.add_argument("--calib", required=True, help="Calibration file")
    parser.add_argument("--csv_out", required=True, help="Output CSV path")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    arch = GPUArchitecture(device_id=0, calibration_file=args.calib)
    vector_sizes = [int(s) for s in args.vector_sizes.split(",")] if args.vector_sizes else [1024, 8192, 65536, 1048576]

    with open(args.kernel_file) as f:
        kernel_src = f.read()

    all_results = []
    for n in vector_sizes:
        print(f"Processing vector_size={n}...")
        results = run_configuration(
            args.kernel_file,
            kernel_src, arch, n, args.iterations
        )
        all_results.extend(results)

    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    columns = [
        "vector_size", "block_x", "thread_count",
        "predicted_time_ns", "actual_time_ns",
        "predicted_power", "actual_power",
        "mem_coal", "mem_uncoal", "mem_partial",
        "local_insts", "shared_insts", "synch_insts",
        "fp_insts", "int_insts", "sfu_insts", "alu_insts",
        "total_insts", "registers_per_thread", "shared_mem_bytes"
    ]
    df = df[columns]

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    # Save DataFrame to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"energy_model_results_{timestamp}.csv"
    csv_path = os.path.join(os.path.dirname(args.csv_out), csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()