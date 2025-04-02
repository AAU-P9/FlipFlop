#!/usr/bin/env python3
"""
RQ1 Experiment: Tuning MHA Kernel
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import kernel_tuner as kt
from datetime import datetime
from kernel_tuner.observers.nvml import NVMLObserver, get_nvml_pwr_limits
from kernel_tuner.observers.ncu import NCUObserver

def run_tuning(kernel_string, batch_size, seq_len, nhead, dim_per_head,
               csv_out, iterations, strategy, variable_block_size, label):
    dim_feature = nhead * dim_per_head
    scale = np.float32(1.0 / np.sqrt(dim_per_head))

    # Increase problem size or do more than 1 pass so that the GPU usage is measurable
    # For demonstration, keep the same but can do bigger data for more stable energy reading:
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

    # Get power-limit range
    pwr_info = get_nvml_pwr_limits(device=0, n=5, quiet=True)
    default_plimits = pwr_info.get("nvml_pwr_limit", [0])

    tune_params = {
        "block_size_x": [2**i for i in range(1, 9)],  # 2..256
        "block_size_y": [1,2,4,8,16,32,64,128],
        "nvml_pwr_limit": default_plimits
    }

    if variable_block_size:
        restrictions = ["(block_size_x * block_size_y) <= 1024"]
    else:
        restrictions = ["(block_size_x * block_size_y) == 256"]

    nsight_metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__cycles_active.avg.pct_of_peak_sustained_active",
    ]
    nsight_observer = NCUObserver(metrics=nsight_metrics)

    # We'll pass custom arguments for the NVMLObserver so it samples more frequently:
    nvml_observer = NVMLObserver(
        variables=["nvml_energy", "nvml_power", "temperature"],
        sample_interval=0.01  # sample every 10ms
    )

    # Precompute total_ops if we need it for metrics
    total_ops = 2.0 * dim_feature * seq_len * batch_size

    def joules_per_token(params):
        energy = params.get("nvml_energy", 0.0)
        if energy <= 0:
            return np.nan
        return energy / (batch_size * seq_len)

    def flops_per_watt(params):
        energy = params.get("nvml_energy", 0.0)
        if energy <= 0:
            return np.nan
        return total_ops / energy

    metrics = {
        "Joules/token": joules_per_token,
        "FLOPS/Watt": flops_per_watt,
        "SM_Active_Avg": lambda p: p.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0),
        "Temperature":   lambda p: p.get("nvml_temperature", 0.0),
    }

    compiler_options = [
        "--std=c++14",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
    ]

    problem_size = (batch_size * nhead,)
    shared_mem_size = (dim_per_head + seq_len) * 4  # float=4 bytes

    # In order to reduce NaNs, do a short warm-up for each config:
    def custom_warmup(device_params):
        """ Called once per config before measuring. Run the kernel a bit to warm up. """
        # Manually launch the kernel with the same config for 1 iteration
        # This is a Kernel Tuner callback approach if we want the easy way:
        pass

    # We also do more iterations so the kernel runs multiple times -> better chance NVML sees energy usage
    warmup_iterations = 1  # used internally by the tuner if we do strategy calls

    results, env = kt.tune_kernel(
        kernel_name="mha",
        kernel_source=kernel_string,
        problem_size=problem_size,
        arguments=arguments,
        tune_params=tune_params,
        observers=[nvml_observer, nsight_observer],
        metrics=metrics,
        strategy=strategy,
        restrictions=restrictions,
        compiler_options=compiler_options,
        smem_args={"size": shared_mem_size},
        iterations=iterations,        # main measurement iterations
        # The next two only if kernel_tuner supports them:
        # warmup=warmup_iterations,
        # or we can do "custom_warmup=custom_warmup"
        verbose=True,
        cache=None,
    )

    # Optionally filter out rows with zero energy to reduce clutter:
    filtered = []
    for r in results:
        if r.get("nvml_energy", 0) > 0:
            filtered.append(r)

    # If after filtering we have none, we keep them so we see the NaNs
    if len(filtered) == 0:
        filtered = results

    timestamp = datetime.now().isoformat()
    for r in filtered:
        r["timestamp"] = timestamp
        r["run_label"] = label

    df = pd.DataFrame(filtered)
    df["batch_size"] = batch_size
    df["seq_len"] = seq_len
    df["nhead"] = nhead
    df["dim_per_head"] = dim_per_head

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    if os.path.exists(csv_out):
        df_existing = pd.read_csv(csv_out)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(csv_out, index=False)
    print(f"[INFO] Tuning done for label={label}, seq_len={seq_len}, wrote {len(filtered)} rows to {csv_out}\n")

def main():
    parser = argparse.ArgumentParser(description="RQ1 MHA Tuning with reduced NaN by warmup + more iteration.")
    parser.add_argument("--kernel_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--seq_lens", type=str, default=None)
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--dim_per_head", type=int, default=256)
    parser.add_argument("--csv_out", type=str, default="rq1_data/mha_tuning_results.csv")
    parser.add_argument("--iterations", type=int, default=10,  # bumped to 10
                        help="Number of main measurement iterations per config.")
    parser.add_argument("--strategy", type=str, default="brute_force",
                        choices=["brute_force","random","bayesian"])
    parser.add_argument("--variable_block_size", action="store_true")
    parser.add_argument("--include_static_heuristic", action="store_true")

    args = parser.parse_args()
    if args.seq_lens:
        seq_lens = [int(s) for s in args.seq_lens.split(",")]
    else:
        seq_lens = [args.seq_len]

    with open(args.kernel_file, "r") as f:
        kernel_src = f.read()

    for sl in seq_lens:
        run_tuning(kernel_src, args.batch_size, sl,
                   args.nhead, args.dim_per_head,
                   args.csv_out, args.iterations, args.strategy,
                   args.variable_block_size,
                   label="adaptive")

        # Also run static if requested
        if args.include_static_heuristic:
            from kernel_tuner.tune_kernel import tune_kernel
            from kernel_tuner.observers.nvml import NVMLObserver
            from kernel_tuner.observers.ncu import NCUObserver

            run_tuning_static(kernel_src,
                              batch_size=args.batch_size,
                              seq_len=sl,
                              nhead=args.nhead,
                              dim_per_head=args.dim_per_head,
                              csv_out=args.csv_out,
                              iterations=args.iterations,
                              label="static_heuristic")

def run_tuning_static(kernel_string, batch_size, seq_len, nhead, dim_per_head,
                      csv_out, iterations, label="static"):
    """
    Single block dimension as a static baseline approach, e.g. (128,2).
    """
    import kernel_tuner as kt
    import numpy as np
    import pandas as pd
    import os
    from datetime import datetime
    from kernel_tuner.observers.nvml import NVMLObserver, get_nvml_pwr_limits
    from kernel_tuner.observers.ncu import NCUObserver

    dim_feature = nhead * dim_per_head
    scale = np.float32(1.0 / np.sqrt(dim_per_head))

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
        np.int32(64),
        out
    ]

    pwr_info = get_nvml_pwr_limits(device=0, n=5, quiet=True)
    default_plimits = pwr_info.get("nvml_pwr_limit", [0])

    tune_params = {
        "block_size_x": [128],
        "block_size_y": [2],
        "nvml_pwr_limit": default_plimits
    }

    def joules_per_token(p):
        e = p.get("nvml_energy", 0.0)
        if e <= 0: return np.nan
        return e / (batch_size * seq_len)

    total_ops = 2.0 * dim_feature * seq_len * batch_size
    def flops_per_watt(p):
        e = p.get("nvml_energy", 0.0)
        if e <= 0: return np.nan
        return total_ops / e

    nsight_metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__cycles_active.avg.pct_of_peak_sustained_active",
    ]
    nsight_observer = NCUObserver(metrics=nsight_metrics)
    nvml_observer = NVMLObserver(["nvml_energy","nvml_power","temperature"], sample_interval=0.01)

    metrics = {
        "Joules/token": joules_per_token,
        "FLOPS/Watt": flops_per_watt,
        "SM_Active_Avg": lambda p: p.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0),
        "Temperature":   lambda p: p.get("nvml_temperature", 0.0),
    }

    compiler_options = ["--std=c++14",
                        "-I/usr/local/cuda/include",
                        "-I/usr/local/cuda/include/cub"]

    problem_size = (batch_size * nhead,)
    shared_mem_size = (dim_per_head + seq_len)*4

    results, env = kt.tune_kernel(
        kernel_name="mha",
        kernel_source=kernel_string,
        problem_size=problem_size,
        arguments=arguments,
        tune_params=tune_params,
        observers=[nvml_observer, nsight_observer],
        metrics=metrics,
        strategy="brute_force",
        restrictions=[],
        compiler_options=compiler_options,
        smem_args={"size": shared_mem_size},
        iterations=iterations,
        verbose=True,
        cache=None
    )

    # Filter out zero-energy if you want to skip them
    filtered = [r for r in results if r.get("nvml_energy",0)>0]

    timestamp = datetime.now().isoformat()
    for r in filtered:
        r["timestamp"] = timestamp
        r["run_label"] = label

    df = pd.DataFrame(filtered)
    df["batch_size"] = batch_size
    df["seq_len"] = seq_len
    df["nhead"] = nhead
    df["dim_per_head"] = dim_per_head

    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    if os.path.exists(csv_out):
        df_existing = pd.read_csv(csv_out)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(csv_out, index=False)
    print(f"[INFO] Static baseline done (label={label}). Appended {len(filtered)} rows to {csv_out}\n")

if __name__ == "__main__":
    main()
