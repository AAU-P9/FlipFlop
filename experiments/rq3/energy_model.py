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
    """Generate all valid block size combinations based on restrictions"""
    block_sizes = []
    for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for y in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            total = x * y
            if total >= 32 and total <= 1024 and total % 32 == 0:
                block_sizes.append((x, y))
    return block_sizes

def prepare_kernel_args(batch_size, seq_len, dim_feature, nhead):
    """Generate kernel arguments for given configuration"""
    scale = np.float32(1.0 / np.sqrt(dim_feature // nhead))
    
    q = np.random.randn(batch_size, dim_feature).astype(np.float32)
    k = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32)
    v = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32)
    out = np.zeros_like(q)
    
    return [
        q, k, v,
        np.int32(batch_size),
        np.int32(seq_len),
        np.int32(dim_feature),
        np.int32(dim_feature),
        np.int32(nhead),
        scale,
        np.int32(64),  # threshold
        out,
    ]

def run_configuration(kernel_path , kernel_src, arch, batch_size, seq_len, nhead, dim_per_head, iterations):
    """Run energy model predictions and measurements for all valid block sizes"""
    dim_feature = nhead * dim_per_head
    args = prepare_kernel_args(batch_size, seq_len, dim_feature, nhead)
    block_combos = generate_block_combinations()
    
    results = []
    for block_x, block_y in block_combos:
        # Compile and analyze
        mod, ptx_str, ptxas_log, kname = compile_kernel(kernel_path, arch)
        analyzer = PTXAnalyzer(ptx_str, ptxas_log, arch, block_x, block_y, {})
        analysis = analyzer.analyze()
        
        # Time prediction
        time_model = HongKimExecutionTimeModel(
            arch, analysis, (batch_size * nhead, 1), (block_x, block_y))
        est_time_ns = time_model.estimate_time_ns()
        
        # Power prediction
        warp_size = arch.attrs.get('WARP_SIZE', 32)
        warps_per_block = (block_x * block_y + warp_size - 1) // warp_size
        blocks_per_sm = time_model._calc_blocks_per_sm(block_x * block_y)
        warps_per_sm = blocks_per_sm * warps_per_block
        
        power_estimator = HongKimPowerEstimator(arch, analysis)
        predicted_power = power_estimator.estimate_power(
            est_time_ns * 1e-9, warps_per_sm, arch.sm_count, arch.clock_rate_hz)
        
        compiler_options = [
        "--std=c++14",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
    ]
        shared_mem_size = (dim_per_head + seq_len) * 4;  # if needed
    
        # Actual measurements
        nvml_observer = NVMLObserver(["nvml_energy", "nvml_power"])
        result, _ = kt.tune_kernel(
            kernel_name=kname,
            kernel_source=kernel_src,
            problem_size=(batch_size * nhead, 1),
            arguments=args,
            tune_params={"block_size_x": [block_x], "block_size_y": [block_y]},
            compiler_options=compiler_options,
            smem_args        = {"size": shared_mem_size},
            observers=[nvml_observer],
            metrics={"time_ns": lambda p: p["time"] * 1e6},
            iterations=iterations,
            verbose=False
        )
        
        # Collect results
        entry = {
            "seq_len": seq_len,
            "block_x": block_x,
            "block_y": block_y,
            "thread_count": block_x * block_y,
            "predicted_time_ns": est_time_ns,
            "predicted_power": predicted_power,
            "actual_time_ns": result[0]["time_ns"],
            "actual_power": np.mean(result[0]["nvml_power"]),
            **analysis.__dict__
        }
        results.append(entry)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="RQ3 Comprehensive Energy Analysis")
    parser.add_argument("--kernel_file", required=True, help="Path to CUDA kernel")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_lens", type=str, help="Comma-separated sequence lengths")
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--dim_per_head", type=int, default=256)
    parser.add_argument("--calib", required=True, help="Calibration file")
    parser.add_argument("--csv_out", required=True, help="Output CSV path")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    arch = GPUArchitecture( device_id = 0, calibration_file=args.calib)
    seq_lens = [int(s) for s in args.seq_lens.split(",")] if args.seq_lens else [128]

    with open(args.kernel_file) as f:
        kernel_src = f.read()

    all_results = []
    for seq_len in seq_lens:
        print(f"Processing seq_len={seq_len}...")
        results = run_configuration( args.kernel_file,
            kernel_src, arch, args.batch_size, seq_len,
            args.nhead, args.dim_per_head, args.iterations
        )
        all_results.extend(results)

    # Create DataFrame and save
    df = pd.DataFrame(all_results)
    df["batch_size"] = args.batch_size
    df["nhead"] = args.nhead
    df["dim_per_head"] = args.dim_per_head
    
    # Reorder columns for better readability
    columns = [
        "seq_len", "batch_size", "nhead", "dim_per_head",
        "block_x", "block_y", "thread_count",
        "predicted_time_ns", "actual_time_ns",
        "predicted_power", "actual_power",
        "mem_coal", "mem_uncoal", "mem_partial",
        "local_insts", "shared_insts", "synch_insts",
        "fp_insts", "int_insts", "sfu_insts", "alu_insts",
        "total_insts", "registers_per_thread", "shared_mem_bytes"
    ]
    df = df[columns]

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    if os.path.exists(args.csv_out):
        existing = pd.read_csv(args.csv_out)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(args.csv_out, index=False)

if __name__ == "__main__":
    main()