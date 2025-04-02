#!/usr/bin/env python3
"""
RQ1 Experiment: Tuning MHA Kernel under the default GPU power limit
====================================================================

Requirements:
 - We test block_size_x,y in the range 1..1024, but restricted so that:
     32 <= block_size_x * block_size_y <= 1024
     (block_size_x * block_size_y) % 32 == 0
 - No power-limit sweeps: we keep the GPU's default power limit.
 - We can choose strategy = "brute_force" or "bayes_opt" from Kernel Tuner.

We measure energy (via NVMLObserver) but do NOT param over power-limit.
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

from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.ncu import NCUObserver

def sm_efficiency(p):
    val_avg = p.get("sm__warps_active.avg.pct_of_peak_sustained_active", np.nan)
    val_max = p.get("sm__warps_active.max.pct_of_peak_sustained_active", np.nan)
    # If val_max is 0 (or NaN), ratio is undefined => return NaN
    if val_max == 0 or np.isnan(val_avg) or np.isnan(val_max):
        return np.nan
    return val_avg / val_max


def run_tuning(kernel_string, batch_size, seq_len, nhead, dim_per_head,
               csv_out, iterations, strategy, label):
    """
    Run kernel tuner for MHA kernel across block_size_x,y pairs where
      32 <= x*y <= 1024 and x*y is multiple of 32,
    measuring energy, throughput, etc. under the default GPU power limit.
    """

    # Setup data
    dim_feature = nhead * dim_per_head
    scale = np.float32(1.0 / np.sqrt(dim_per_head))

    # Random inputs for Q, K, V + output buffer
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
        np.int32(64),  # threshold
        out,
    ]

    # We do NOT param over power-limit. Only the default GPU limit is used.
    # So we skip adding "nvml_pwr_limit" to tune_params.

    # Enumerate block size in range(1..1024), but we rely on restrictions
    tune_params = {
         "block_size_x": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], 
        "block_size_y": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    }

    # Restrict to multiples of 32, up to 1024 total threads
    restrictions = [
        "(block_size_x * block_size_y) >= 32",
        "(block_size_x * block_size_y) <= 1024",
        "(block_size_x * block_size_y) % 32 == 0"
    ]

    # Observers
    # We gather some basic Nsight metrics, plus GPU power from NVML
    nsight_metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.max.pct_of_peak_sustained_active",
        "sm__cycles_active.avg.pct_of_peak_sustained_active",
    ]
    nsight_observer = NCUObserver(metrics=nsight_metrics)
    nvml_observer   = NVMLObserver(["nvml_energy", "nvml_power", "temperature"])

    # We'll compute approximate FLOPS and from that an estimated FLOPS/Watt, etc.
    total_ops = 2.0 * dim_feature * seq_len * batch_size

    # Define derived metrics as an OrderedDict:
    metrics = OrderedDict()
    metrics["Joules/token"] = lambda p: p["nvml_energy"] / (batch_size * seq_len) if p["nvml_energy"] > 0 else np.nan
    metrics["FLOPS/Watt"]   = lambda p: total_ops / p["nvml_energy"] if p["nvml_energy"] > 0 else np.nan
    metrics["SM_Active_Avg"] = lambda p: p.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    metrics["Temperature"]   = lambda p: p.get("temperature", 0.0)
    metrics["SM_Eff"] = sm_efficiency

    compiler_options = [
        "--std=c++14",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
    ]

    problem_size    = (batch_size * nhead,)
    shared_mem_size = (dim_per_head + seq_len) * 4  # if needed

    # Kernel Tuner invocation
    results, env = kt.tune_kernel(
        kernel_name      = "mha",
        kernel_source    = kernel_string,
        problem_size     = problem_size,
        arguments        = arguments,
        tune_params      = tune_params,
        observers        = [nvml_observer, nsight_observer],
        metrics          = metrics,
        strategy         = strategy,      # "brute_force" or "bayes_opt"
        restrictions     = restrictions,
        compiler_options = compiler_options,
        smem_args        = {"size": shared_mem_size},
        iterations       = iterations,
        verbose          = True,
        cache            = f"rq1_data/mha_cache_{label}.json",
        objective        = "Joules/token",           # Primary objective: minimize energy per token
        objective_higher_is_better = False         # Lower Joules/token is better
    )

    # Tag results
    timestamp = datetime.now().isoformat()
    for r in results:
        r["timestamp"] = timestamp
        r["run_label"] = label

    # Convert to DataFrame, add problem details, append to CSV
    df = pd.DataFrame(results)
    df["batch_size"]   = batch_size
    df["seq_len"]      = seq_len
    df["nhead"]        = nhead
    df["dim_per_head"] = dim_per_head

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    if os.path.exists(csv_out):
        existing = pd.read_csv(csv_out)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(csv_out, index=False)

    print(f"[INFO] Done RQ1 Tuning: label={label}, seq_len={seq_len}, wrote {len(results)} rows to {csv_out}.\n")


def main():
    parser = argparse.ArgumentParser(description="RQ1 MHA Tuning - default GPU power, block-size up to 1024 threads.")
    parser.add_argument("--kernel_file", type=str, required=True,
                        help="Path to .cu file containing the MHA kernel.")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--seq_len",      type=int, default=128)
    parser.add_argument("--seq_lens",     type=str, default=None,
                        help="Comma-separated list of sequence lengths for multiple runs.")
    parser.add_argument("--nhead",        type=int, default=16)
    parser.add_argument("--dim_per_head", type=int, default=256)
    parser.add_argument("--csv_out",      type=str, default="rq1_data/mha_tuning_results.csv")
    parser.add_argument("--iterations",   type=int, default=5,
                        help="Number of measurement iterations per block config.")
    parser.add_argument("--strategy",     type=str, default="brute_force",
                        choices=["brute_force", "random", "bayes_opt", "genetic", "mcmc"],
                        help="Search strategy used by Kernel Tuner.")
    parser.add_argument("--label",        type=str, default="rq1_adaptive",
                        help="Label to store in the CSV for this run.")
    args = parser.parse_args()

    if args.seq_lens:
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
    else:
        seq_lens = [args.seq_len]

    # Load kernel code
    with open(args.kernel_file, "r") as f:
        kernel_src = f.read()

    for sl in seq_lens:
        run_tuning(
            kernel_string    = kernel_src,
            batch_size       = args.batch_size,
            seq_len          = sl,
            nhead            = args.nhead,
            dim_per_head     = args.dim_per_head,
            csv_out          = args.csv_out,
            iterations       = args.iterations,
            strategy         = args.strategy,
            label            = args.label
        )


if __name__ == "__main__":
    main()
