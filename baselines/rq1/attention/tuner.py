#!/usr/bin/env python3
"""
Tune MHA kernel variants with block/warp reductions for energy efficiency
"""
import numpy as np
import kernel_tuner as kt
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.ncu import NCUObserver
from kernel_tuner.observers.nvml import get_nvml_pwr_limits, get_nvml_gr_clocks, get_nvml_mem_clocks, get_idle_power
import os

import pandas as pd



def tune_mha_variants():
    # Kernel variants to tune
    kernels = [
        "kernel1_blockReduce", "kernel1_warpReduce",
        "kernel2_blockReduce", "kernel2_warpReduce",
        "kernel3"
    ]

    # Unified tuning parameters for all variants
    tune_params = {
        "block_size_x": [32, 64, 128, 256, 512, 1024],
        "nvml_pwr_limit": get_nvml_pwr_limits(device=0)["nvml_pwr_limit"],
        # "kernel_type": list(kernels.keys())
    }


    # Problem dimensions
    n = 1024  # Number of query vectors
    d = 768    # Dimension per head
    seq_len = 512

    # Generate synthetic data matching kernel signatures
    key = np.random.randn(n * d).astype(np.float32)
    query = np.random.randn(d).astype(np.float32)
    value = np.random.randn(n * seq_len * d).astype(np.float32)
    dot_product = np.zeros(n, dtype=np.float32)
    exp_sum = np.zeros(1, dtype=np.float32)
    output = np.zeros(d, dtype=np.float32)

    # Common arguments for all kernels
    # common_args = [
    #     key, query, dot_product, exp_sum,
    #     value, output, np.int32(n), np.int32(d), np.int32(seq_len)
    # ]

   

    # Energy metrics
    metrics = {
        "Joules/token": lambda p: p["nvml_energy"] / (n * seq_len),
        "FLOPS/Watt": lambda p: (2 * n * d * seq_len) / p["nvml_energy"]
    }

    # Configure observers
    nvml_observer = NVMLObserver(["nvml_energy", "nvml_power"], continuous_duration = 10)
    ncu_observer = NCUObserver(metrics=[
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
    ])

    folder_name = os.path.basename(os.getcwd())
    with open(os.path.join("/home/srajput/flipflop/cuda_kernel_energy_empirical/baselines/rq1", folder_name, "kernels.h"), "r") as f:
        kernel_code = f.read()

    for kernel_name in kernels:
        # Define problem size based on the kernel's operation
        problem_size = {
            "kernel1_blockReduce": (n, 1),       # 1 block per query vector
            "kernel1_warpReduce": (n//32, 1),     # 32 threads per warp
            "kernel2_blockReduce": (d, 1),       # 1 block per feature
            "kernel2_warpReduce": (d//32, 1),    # Warp-based reduction
            "kernel3": (d, 1)
        }.get(kernel_name, (1, 1))


        base_args = [key, query, dot_product, exp_sum, value, output]
        if "kernel1" in kernel_name:
            args = base_args + [np.int32(n), np.int32(d)]
        elif "kernel2" in kernel_name:
            args = base_args + [np.int32(n), np.int32(d)]
        elif "kernel3" in kernel_name:
            args = [score, value, output, np.int32(n), np.int32(d)]

        # Restrictions for valid configurations
        # For blockReduce kernels using CUB
        restrictions = [
            "block_size_x <= 256" if "blockReduce" in kernel_name else "block_size_x <= 512",
            "block_size_x % 32 == 0",
            f"block_size_x <= {d if 'kernel2' in kernel_name else n}"
        ]


        result, _ = kt.tune_kernel(
            kernel_name=kernel_name,
            kernel_source=kernel_code,
            problem_size=problem_size,
            arguments=args,
            tune_params=tune_params,
            observers=[nvml_observer, ncu_observer],
            metrics=metrics,
            restrictions=restrictions,
            verbose=True,
            compiler_options=[
                "-I/usr/local/cuda/include",
                "-I/usr/local/cuda/include/cub",
                "--expt-relaxed-constexpr",
                "--maxrregcount=64",  # Limit registers per thread
                "-Xptxas=-v"  # Verify register usage
            ],
            cache=os.path.join("/home/srajput/flipflop/cuda_kernel_energy_empirical/rq1_data", folder_name, f"{kernel_name}_cache.json")
        )

if __name__ == "__main__":
    tune_mha_variants()
