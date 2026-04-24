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

from experiments.rq2.gpu_common import GPUArchitecture, compile_kernel
from experiments.rq2.time_model import HongKimExecutionTimeModel
from experiments.rq2.power_model import HongKimPowerEstimator
from experiments.rq2.PTXAnalyzer import PTXAnalyzer

##############################################################################
#  Adapting for llama3.cu-based Multi-Head Attention
##############################################################################

def generate_block_combinations():
    """Generate all valid block size combinations based on restrictions"""
    block_sizes = []
    for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        for y in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            total = x * y
            if total >= 32 and total <= 1024 and total % 32 == 0:
                block_sizes.append((x, y))
    return block_sizes


def prepare_kernel_args(beamsize, n_steps, dim_feature, nhead):
    """
    Build the exact argument list for multi_head_attention_kernel, which expects:
      1) pos (int)
      2) seq_len (int)
      3) sq (float*)
      4) satt (float*)
      5) sxb (float*)
      6) key_cache (float*)
      7) value_cache (float*)
      8) kv_dim (int)
      9) kv_mul (int)
      10) head_size (int)
      11) loff (int)

    We choose some dummy defaults for pos=0, loff=0, etc.
    Adjust shapes as needed for your real scenario.
    """
    # Suppose pos=0 (the current decoding position).
    pos = np.int32(0)

    # Use seq_len = n_steps in the kernel
    seq_len = np.int32(n_steps)

    # We define "dim" = nhead*dim_per_head to match llama3's "p->dim".
    # We'll treat kv_dim = dim for a 1:1 Q/K shape, kv_mul=1, etc.
    dim = dim_feature
    head_size = dim // nhead

    # Create arrays for sq, satt, sxb, key_cache, value_cache
    # sq is a single [dim] for the "query" vector (like s->q in your code).
    sq = np.random.randn(dim).astype(np.float32)

    # satt is a single [seq_len*nheads], but the kernel uses only [n_steps*nheads].
    # For simplicity, we do [nhead * n_steps].
    # But the kernel sees pointer "satt + h*seq_len". We'll flatten it:
    satt = np.zeros((nhead * n_steps,), dtype=np.float32)

    # sxb is also [dim]
    sxb = np.zeros(dim, dtype=np.float32)

    # key_cache and value_cache: we can do [n_steps * dim], 
    # flatten them as 1D. The kernel indexes them with "t * kv_dim + (h/kv_mul)*head_size" etc.
    # For a single layer example, set n_layers=1 => loff=0
    key_cache = np.random.randn(n_steps, dim).astype(np.float32).ravel()
    value_cache = np.random.randn(n_steps, dim).astype(np.float32).ravel()

    # kv_dim = dim, kv_mul=1, head_size, loff=0
    kv_dim = np.int32(dim)
    kv_mul = np.int32(1)
    head_size_int = np.int32(head_size)
    loff = np.int32(0)

    # Return in exact order matching the kernel signature
    return [
        pos,           # 1) int
        seq_len,       # 2) int
        sq,            # 3) float*
        satt,          # 4) float*
        sxb,           # 5) float*
        key_cache,     # 6) float*
        value_cache,   # 7) float*
        kv_dim,        # 8) int
        kv_mul,        # 9) int
        head_size_int, # 10) int
        loff           # 11) int
    ]


def run_configuration(kernel_path, kernel_src, arch,
                      beamsize, n_steps, nhead, dim_per_head, iterations):
    """
    Compile and run 'llama3.cu' for all valid block sizes, collecting
    actual and predicted time & power data.
    """
    dim_feature = nhead * dim_per_head
    args = prepare_kernel_args(beamsize, n_steps, dim_feature, nhead)
    block_combos = generate_block_combinations()

    results = []
    for block_x, block_y in block_combos:
        # 1) Compile and analyze
        mod, ptx_str, ptxas_log, kname = compile_kernel(kernel_path, arch)
        analyzer = PTXAnalyzer(ptx_str, ptxas_log, arch, block_x, block_y, {})
        analysis = analyzer.analyze()

        # 2) Predict time via Hong–Kim
        time_model = HongKimExecutionTimeModel(arch, analysis,
                                               (beamsize, 1),
                                               (block_x, block_y))
        est_time_ns = time_model.estimate_time_ns()

        # 3) Predict power using Hong–Kim
        warp_size = arch.attrs.get('WARP_SIZE', 32)
        warps_per_block = (block_x * block_y + warp_size - 1) // warp_size
        blocks_per_sm = time_model._calc_blocks_per_sm(block_x * block_y)
        warps_per_sm = blocks_per_sm * warps_per_block

        power_estimator = HongKimPowerEstimator(arch, analysis)
        predicted_power = power_estimator.estimate_power(
            (est_time_ns * arch.clock_rate_hz) / 1e9,
            warps_per_sm, arch.sm_count, arch.clock_rate_hz
        )

        compiler_options = [
            "--std=c++14",
            "-I/usr/local/cuda/include",
            "-I/usr/local/cuda/include/cub",
        ]
        # If llama3.cu uses shared memory differently, adapt here
        shared_mem_size = (dim_per_head + n_steps) * 4  # example

        # 4) Actual measurements
        nvml_observer = NVMLObserver(["nvml_energy", "nvml_power"])
        result, _ = kt.tune_kernel(
            kernel_name="multi_head_attention_kernel",
            kernel_source=kernel_src,
            problem_size=(beamsize * nhead, 1),  # or (beamsize * nhead, 1)
            arguments=args,
            tune_params={"block_size_x": [block_x], "block_size_y": [block_y]},
            compiler_options=compiler_options,
            smem_args={"size": shared_mem_size},
            observers=[nvml_observer],
            metrics={"time_ns": lambda p: p["time"] * 1e6},  # sec->ns
            iterations=iterations,
            verbose=False
        )

        # print result and keys
        print("results is:", result[0])

        # 5) Collect results
        entry = {
            "n_steps": n_steps,
            "block_x": block_x,
            "block_y": block_y,
            "thread_count": block_x * block_y,
            "predicted_time_ns": est_time_ns,
            "predicted_power": predicted_power,
            "actual_time_ns": result[0]["time_ns"],
            "actual_power": float(np.mean(result[0]["nvml_power"])),
            # PTX analysis breakdown
            "mem_coal": analysis.mem_coal,
            "mem_uncoal": analysis.mem_uncoal,
            "mem_partial": analysis.mem_partial,
            "local_insts": analysis.local_insts,
            "shared_insts": analysis.shared_insts,
            "synch_insts": analysis.synch_insts,
            "fp_insts": analysis.fp_insts,
            "int_insts": analysis.int_insts,
            "sfu_insts": analysis.sfu_insts,
            "alu_insts": analysis.alu_insts,
            "total_insts": analysis.total_insts,
            "registers_per_thread": analysis.registers_per_thread,
            "shared_mem_bytes": analysis.shared_mem_bytes,
        }
        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser(description="RQ4 Energy Analysis for llama3 MHA")
    parser.add_argument("--kernel_file", required=True, help="Path to llama3.cu")
    parser.add_argument("--beamsize", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=9)        # example
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--dim_per_head", type=int, default=256)
    parser.add_argument("--calib", required=True, help="Calibration file")
    parser.add_argument("--csv_out", required=True, help="Output CSV path")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    # Load GPU architecture config
    arch = GPUArchitecture(device_id=0, calibration_file=args.calib)

    # Read the kernel source
    with open(args.kernel_file) as f:
        kernel_src = f.read()

    # print("kernel src:", kernel_src)

    # -----------
    # Prepare to sweep block_x, block_y
    # -----------
    # Possibly we can also do multiple n_steps if desired. 
    # For now, just run one n_steps from command line

    results = run_configuration(
        kernel_path=args.kernel_file,
        kernel_src=kernel_src,
        arch=arch,
        beamsize=args.beamsize,
        n_steps=args.n_steps,
        nhead=args.nhead,
        dim_per_head=args.dim_per_head,
        iterations=args.iterations
    )

    # Put results in DataFrame
    df = pd.DataFrame(results)
    df["beamsize"] = args.beamsize
    df["nhead"] = args.nhead
    df["dim_per_head"] = args.dim_per_head

    columns = [
        "n_steps", "beamsize", "nhead", "dim_per_head",
        "block_x", "block_y", "thread_count",
        "predicted_time_ns", "actual_time_ns",
        "predicted_power", "actual_power",
        "mem_coal", "mem_uncoal", "mem_partial",
        "local_insts", "shared_insts", "synch_insts",
        "fp_insts", "int_insts", "sfu_insts", "alu_insts",
        "total_insts", "registers_per_thread", "shared_mem_bytes"
    ]
    df = df[columns]

    # Save CSV
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"llama3_energy_results_{timestamp}.csv"
    csv_path = os.path.join(os.path.dirname(args.csv_out), csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to {csv_path}")


if __name__ == "__main__":
    main()
