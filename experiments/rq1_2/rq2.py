#!/usr/bin/env python3
"""
RQ2 Experiment: Tuning MHA Kernel with Real-time Power Limit Adaptation
===========================================================================
This experiment investigates whether jointly adapting the GPU power limit
(and hence enforcing dynamic power capping via nvidia-smi) and the thread-block
dimensions in real time can reduce total decoding energy in LLM inference.

Methodology:
  - Decoding Scenario: The MHA kernel is repeatedly launched with different
    sequence lengths and batch sizes to simulate LLM decoding.
  - Power-limit Variation: We vary the "nvml_pwr_limit" parameter using the 10 power limits
    returned by get_nvml_pwr_limits.
  - Block-dimension Adaptation: All (block_size_x, block_size_y) pairs are
    considered such that 32 <= (block_size_x * block_size_y) <= 1024 and is a multiple of 32.
  - Bayesian Optimization: You can choose between a full brute-force sweep or a
    “bayesian” strategy using Kernel Tuner.
    
Results (energy in Joules/token, FLOPS/Watt, occupancy, temperature)
are stored to a CSV file whose name is appended with the strategy.
"""

import argparse
import os
import numpy as np
import pandas as pd
import kernel_tuner as kt
from datetime import datetime
import pycuda.driver as pcudadriver
import pycuda.autoinit
from collections import OrderedDict

from kernel_tuner.observers.nvml import NVMLObserver, get_nvml_pwr_limits
from kernel_tuner.observers.ncu import NCUObserver

def run_tuning(kernel_string, batch_size, seq_len, nhead, dim_per_head,
               csv_out, iterations, strategy, label):
    """
    Tuning routine for RQ2:
      - Varies block configuration (x,y) where total threads ∈ [32, 1024] and divisible by 32.
      - Varies GPU power limits using the 10 values returned by get_nvml_pwr_limits.
    """
    dim_feature = nhead * dim_per_head
    scale = np.float32(1.0 / np.sqrt(dim_per_head))

    # Prepare input data for Q, K, V and output buffer.
    q = np.random.randn(batch_size, dim_feature).astype(np.float32).ravel()
    k = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32).ravel()
    v = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32).ravel()
    out = np.zeros_like(q)

    arguments = [
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

    # Tuning parameters now include block_size_x, block_size_y, and nvml_pwr_limit.
    tune_params = {
        "block_size_x": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "block_size_y": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        "nvml_pwr_limit": get_nvml_pwr_limits(device=0, n=10, quiet=False)["nvml_pwr_limit"]
    }

    # Restrict to valid block configurations:
    #   total threads must be at least 32, at most 1024, and a multiple of 32.
    restrictions = [
        "(block_size_x * block_size_y) >= 32",
        "(block_size_x * block_size_y) <= 1024",
        "(block_size_x * block_size_y) % 32 == 0"
    ]

    # Observers: NVML for power/energy and Nsight Compute for SM metrics.
    nsight_metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.max.pct_of_peak_sustained_active",
        "sm__cycles_active.avg.pct_of_peak_sustained_active",
    ]
    nsight_observer = NCUObserver(metrics=nsight_metrics)
    nvml_observer   = NVMLObserver(["nvml_energy", "nvml_power", "temperature"])

    # Calculate derived metrics: total_ops.
    total_ops = 2.0 * dim_feature * seq_len * batch_size

    # Define metrics inline, without using separate safe_* functions.
    metrics = OrderedDict()
    # Joules/token: if nvml_energy <= 0, return a high penalty (e.g., 1e6)
    metrics["Joules/token"] = lambda p: (p.get("nvml_energy", 0.0) / (batch_size * seq_len)) if (p.get("nvml_energy", 0.0) > 0) else 1e6
    # FLOPS/Watt: if nvml_energy <= 0, return 0
    metrics["FLOPS/Watt"] = lambda p: (total_ops / p.get("nvml_energy", 0.0)) if (p.get("nvml_energy", 0.0) > 0) else 0.0
    # SM Active Average Utilization:
    metrics["SM_Active_Avg"] = lambda p: p.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    # Temperature:
    metrics["Temperature"] = lambda p: p.get("temperature", 0.0)
    # SM Efficiency: ratio of avg to max; if max is zero, return 0
    # metrics["SM_Eff"] = lambda p: (p.get("sm__warps_active.avg.pct_of_peak_sustained_active", np.nan) / p.get("sm__warps_active.max.pct_of_peak_sustained_active", np.nan)) if (p.get("sm__warps_active.max.pct_of_peak_sustained_active", 0) not in [0, None] and not np.isnan(p.get("sm__warps_active.avg.pct_of_peak_sustained_active", np.nan))) else 0.0

    compiler_options = [
        "--std=c++14",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
    ]

    problem_size    = (batch_size * nhead,)
    shared_mem_size = (dim_per_head + seq_len) * 4  # in bytes

    # Kernel Tuner call.
    results, env = kt.tune_kernel(
        kernel_name      = "mha",
        kernel_source    = kernel_string,
        problem_size     = problem_size,
        arguments        = arguments,
        tune_params      = tune_params,
        observers        = [nvml_observer, nsight_observer],
        metrics          = metrics,
        strategy         = strategy,  # "brute_force" or "bayes_opt"
        restrictions     = restrictions,
        compiler_options = compiler_options,
        smem_args        = {"size": shared_mem_size},
        iterations       = iterations,
        verbose          = True,
        cache            = f"rq2_data/mha_cache_{label}.json",
        objective        = "Joules/token", 
        objective_higher_is_better = False
    )

    # Tag each result with timestamp and run label.
    timestamp = datetime.now().isoformat()
    for r in results:
        r["timestamp"] = timestamp
        r["run_label"] = label

    df = pd.DataFrame(results)
    df["batch_size"] = batch_size
    df["seq_len"] = seq_len
    df["nhead"] = nhead
    df["dim_per_head"] = dim_per_head

    # Append results to CSV. Auto-append the strategy to the CSV file name.
    base, ext = os.path.splitext(csv_out)
    csv_filename = f"{base}_{strategy}{ext}"
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    if os.path.exists(csv_filename):
        existing_df = pd.read_csv(csv_filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(csv_filename, index=False)

    print(f"[INFO] RQ2 Tuning done for label={label}, seq_len={seq_len}, strategy={strategy}, wrote {len(results)} rows to {csv_filename}\n")

def main():
    parser = argparse.ArgumentParser(description="RQ2 MHA Tuning with Real-time Power Limit Adaptation.")
    parser.add_argument("--kernel_file", type=str, required=True,
                        help="Path to the .cu file containing the MHA kernel.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seq_lens", type=str, default=None,
                        help="Comma-separated sequence lengths (e.g., '128,256,512').")
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--dim_per_head", type=int, default=256)
    parser.add_argument("--csv_out", type=str, default="rq2_data/mha_tuning_results.csv",
                        help="Base CSV output filename; strategy name is appended automatically.")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of measurement iterations per configuration.")
    parser.add_argument("--strategy", type=str, default="brute_force",
                        help="Kernel Tuner search strategy (brute_force or bayesian).")
    parser.add_argument("--label", type=str, default="rq2_adaptive",
                        help="Run label to tag this tuning experiment.")
    args = parser.parse_args()

    if args.seq_lens:
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
    else:
        seq_lens = [args.seq_len]

    # Load kernel source code.
    with open(args.kernel_file, "r") as f:
        kernel_src = f.read()

    # Run tuning for each specified sequence length.
    for sl in seq_lens:
        run_tuning(
            kernel_string = kernel_src,
            batch_size = args.batch_size,
            seq_len = sl,
            nhead = args.nhead,
            dim_per_head = args.dim_per_head,
            csv_out = args.csv_out,
            iterations = args.iterations,
            strategy = args.strategy,
            label = args.label
        )

if __name__ == "__main__":
    main()
