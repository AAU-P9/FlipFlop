#!/usr/bin/env python3
"""
RQ4 Experiment: Energy Efficiency Analysis for MHA Kernel Block Shapes
===========================================================================
This experiment measures actual energy consumption across different thread-block
configurations for the llama3.cu multi-head attention kernel.
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import kernel_tuner as kt
from kernel_tuner.observers.nvml import NVMLObserver, get_nvml_pwr_limits
from kernel_tuner.observers.ncu import NCUObserver

import pycuda.driver as pcudadriver
import pycuda.autoinit

def prepare_kernel_args(batch_size, seq_len, nhead, dim_per_head):
    """Prepare input data for MHA kernel"""
    dim_feature = nhead * dim_per_head
    scale = np.float32(1.0 / np.sqrt(dim_per_head))

    q = np.random.randn(batch_size, dim_feature).astype(np.float32).ravel()
    k = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32).ravel()
    v = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32).ravel()
    out = np.zeros_like(q)

    return [
        q, k, v,
        np.int32(batch_size),
        np.int32(seq_len),
        np.int32(dim_feature),
        np.int32(dim_feature),
        np.int32(nhead),
        scale,
        np.int32(64),  # threshold parameter
        out,
    ]

def run_experiment(kernel_src, batch_size, seq_len, nhead, dim_per_head, csv_out, iterations, label):
    """Main experiment routine measuring energy across block configurations"""
    # Prepare kernel arguments and problem size
    args = prepare_kernel_args(batch_size, seq_len, nhead, dim_per_head)
    problem_size = (batch_size * nhead,)
    dim_feature = nhead * dim_per_head
    total_ops = 2.0 * dim_feature * seq_len * batch_size

    # Define tuning parameters and restrictions
    tune_params = {
        "block_size_x": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "block_size_y": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "nvml_pwr_limit": get_nvml_pwr_limits(device=0, n=10, quiet=False)["nvml_pwr_limit"]
    }
    
    restrictions = [
        "(block_size_x * block_size_y) >= 32",
        "(block_size_x * block_size_y) <= 1024",
        "(block_size_x * block_size_y) % 32 == 0"
    ]

    # Configure observers and metrics
    nsight_metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__sass_inst_executed_op_shared_ld.avg.per_cycle_active",
        # "dram__bytes_read.sum.per_second"
    ]
    observers = [
        NVMLObserver(["nvml_energy", "nvml_power"]),
        NCUObserver(nsight_metrics)
    ]

    metrics = OrderedDict()
    metrics["Joules/token"] = lambda p: p["nvml_energy"] / (batch_size * seq_len)
    # metrics["FLOPS/Watt"] = lambda p: total_ops / p["nvml_energy"]
    # metrics["DRAM_BW_GBs"] = lambda p: p.get("dram__bytes_read.sum.per_second", 0) / 1e9
    metrics["SM_Utilization"] = lambda p: p.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)


    compiler_options = [
        "--std=c++14",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
    ]

    shared_mem_size = (dim_per_head + seq_len) * 4  # in bytes

    # Kernel Tuner configuration
    results, _ = kt.tune_kernel(
        kernel_name="multi_head_attention_kernel",
        kernel_source=kernel_src,
        problem_size=problem_size,
        arguments=args,
        tune_params=tune_params,
        observers=observers,
        metrics=metrics,
        restrictions=restrictions,
        compiler_options=compiler_options,
        smem_args = {"size": shared_mem_size},
        iterations=iterations,
        verbose=True,
        objective="Joules/token",
        objective_higher_is_better=False
    )

    # Add metadata and save results
    timestamp = datetime.now().isoformat()
    for r in results:
        r.update({
            "timestamp": timestamp,
            "label": label,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "nhead": nhead,
            "dim_per_head": dim_per_head
        })

    df = pd.DataFrame(results)
    if os.path.exists(csv_out):
        existing_df = pd.read_csv(csv_out)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_out, index=False)
    print(f"Saved {len(results)} configurations to {csv_out}")

def main():
    parser = argparse.ArgumentParser(description="RQ4 Energy Analysis for MHA Block Shapes")
    parser.add_argument("--kernel_file", required=True, help="Path to llama3.cu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--dim_per_head", type=int, default=256)
    parser.add_argument("--csv_out", default="rq4_results.csv", help="Output CSV file")
    parser.add_argument("--iterations", type=int, default=7, help="Measurement iterations per config")
    parser.add_argument("--label", default="baseline", help="Experiment label")
    
    args = parser.parse_args()
    
    with open(args.kernel_file) as f:
        kernel_src = f.read()

    run_experiment(
        kernel_src=kernel_src,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        nhead=args.nhead,
        dim_per_head=args.dim_per_head,
        csv_out=args.csv_out,
        iterations=args.iterations,
        label=args.label
    )

if __name__ == "__main__":
    main()