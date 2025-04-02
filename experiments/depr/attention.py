#!/usr/bin/env python3
"""
Tune MHA kernel block dimensions for energy efficiency and SM occupancy
"""
import numpy as np
import kernel_tuner as kt
from kernel_tuner import run_kernel
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.ncu import NCUObserver
from kernel_tuner.observers.pmt import PMTObserver
from kernel_tuner.observers.nvml import get_nvml_pwr_limits, get_nvml_gr_clocks, get_nvml_mem_clocks, get_idle_power
import pandas as pd
import matplotlib.pyplot as plt



def tune_mha_kernel():

    # Tuning parameters from RQ1 methodology
    tune_params = { 
        "block_size_x": [1, 2, 4, 8, 16, 32, 64, 128, 256], 
        "block_size_y": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        # "seq_len": [256, 512, 1024],  #  sequence length parameter
        "nvml_pwr_limit" : get_nvml_pwr_limits(device=0, n=10, quiet=False)["nvml_pwr_limit"]
    }

    with open("baselines/rq1/main.cu", "r") as f:
        mha_kernel_string = f.read()

    # Problem parameters from your main()
    beam_size = 4
    nhead = 16
    dim_feature = nhead * 256
    # n_steps = max(tune_params["seq_len"])
    n_steps = 9
    THRESHOLD = 64
    scaler = np.sqrt(nhead * 1.0 / dim_feature).astype(np.float32)

    # Generate synthetic data
    q = np.random.randn(beam_size * dim_feature).astype(np.float32)
    k = np.random.randn(beam_size * dim_feature * n_steps).astype(np.float32)
    v = np.random.randn(beam_size * dim_feature * n_steps).astype(np.float32)
    dst = np.zeros_like(q)

    arguments = [q, k, v, np.int32(beam_size), np.int32(n_steps),
                 np.int32(dim_feature), np.int32(dim_feature),
                 np.int32(nhead), scaler, dst]

    # Energy and occupancy metrics
    # metrics = {
    #     "Joules/token": lambda p: p["nvml_energy"] / (beam_size * n_steps),
    #     "FLOPS/Watt": lambda p: (2 * beam_size * n_steps**2 * dim_feature) / (p["nvml_energy"])
    # }

    metrics = {
        "Joules/token": lambda p: p["nvml_energy"] / (beam_size * n_steps),
        "FLOPS/Watt": lambda p: (2 * nhead * n_steps**2 * dim_feature) / p["nvml_energy"],
        "SM_Eff": lambda p: p["sm__warps_active.avg.pct_of_peak_sustained_active"] / 
                           p["sm__warps_active.max.pct_of_peak_sustained_active"]
    }

    # Enhanced observers with detailed metrics
    nsight_metrics = [
        # Core utilization metrics
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.max.pct_of_peak_sustained_active",
        "sm__cycles_active.avg.pct_of_peak_sustained_active",
        
        # # Memory subsystem metrics
        # "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        # "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        # "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
        
        # # Instruction mix analysis
        # "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        # "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        
        # # Power-related metrics
        # "dram__bytes.sum.per_second",
        # "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
        
        # # Pipeline stall analysis
        # "stall_memory_throttle",
        # "stall_exec_dependency"
    ]

    nsight_observer = NCUObserver(metrics=nsight_metrics)
    nvml_observer = NVMLObserver(["nvml_energy", "nvml_power", "temperature"], continuous_duration=0.001)


    restrictions = [
        "block_size_x * block_size_y % 32 == 0", 
        "block_size_x * block_size_y <= 1024",
        # f"(({dim_feature} // {nhead}) % block_size_x) == 0",
        # f"(({dim_feature}//{nhead} + {n_steps}) * 4) <= 48*1024"
    ]

    compiler_options = [
        "-DTHRESHOLD=64",
        "--std=c++11",
        # "--extended-lambda",
        # "--expt-relaxed-constexpr",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
    ]

    # Tuning configuration
    results, _ = kt.tune_kernel(
        kernel_name="mha",
        kernel_source=mha_kernel_string,
        problem_size=(nhead * beam_size, ),
        arguments=arguments,
        tune_params=tune_params,
        observers=[nvml_observer, nsight_observer],
        metrics=metrics,
        iterations=100,
        verbose=True,
        restrictions=restrictions,
        compiler_options=compiler_options,
        strategy="brute_force",
        cache="rq1_data/mha_cache.json",
        # smem_args=lambda p: {"shared_size": ((dim_feature//nhead) + p["seq_len"]) * 4},
        # strategy_options={"max_fevals": 500}
    )

    # Post-processing for Pareto analysis
    df = pd.DataFrame(results)
    df["SM_Occupancy"] = df["sm__warps_active.avg.pct_of_peak_sustained_active"]

    # In tune_mha_kernel() after creating the DataFrame:
    df["num_heads"] = nhead
    df["dim_feature"] = dim_feature
    df["beam_size"] = beam_size

    
    # # Pareto frontier identification
    # pareto_front = []
    # for _, row in df.iterrows():
    #     if not any((x["Joules/token"] <= row["Joules/token"] and 
    #                 x["FLOPS/Watt"] >= row["FLOPS/Watt"]) for x in pareto_front):
    #         pareto_front.append(row.to_dict())
    
    # Save results for visualization
    df.to_csv("rq1_data/mha_tuning_results.csv")
    
    # return pareto_front

if __name__ == "__main__":
    # Validate kernel correctness first

    

    # reference_result = run_kernel("mha", mha_kernel_string, (16,1), arguments, 
    #                             {"block_size_x": 256})
    
    tune_mha_kernel()
    # visualize_pareto(pareto_front)


 
    df = pd.read_csv("rq1_data/mha_tuning_results.csv")

    # Filter configurations meeting RQ1 criteria
    optimal_configs = df[
        (df['FLOPS/Watt'] >= 0.9*df['FLOPS/Watt'].max()) &
        (df['Joules/token'] <= 1.1*df['Joules/token'].min())
    ]

    print("Optimal Configurations:")
    print(optimal_configs[['block_size_x', 'SM_Occupancy', 'Joules/token']])

