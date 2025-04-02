import numpy as np
from collections import OrderedDict
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.ncu import NCUObserver
import kernel_tuner as kt
from kernel_tuner.observers.nvml import get_nvml_pwr_limits, get_nvml_gr_clocks, get_nvml_mem_clocks, get_idle_power
from kernel_tuner.searchspace import Searchspace
from kernel_tuner.strategies.bayes_opt import BayesianOptimization

# Define restrictions globally to share between functions
restrictions = [
    "(block_size_x * block_size_y) <= 1024",
    "(block_size_x * block_size_y) % 32 == 0",
    "tile_size_y <= block_size_y"
]


# Problem size and initialization
m, n, k = 4096, 4096, 4096

problem_size = (m, n)


ARGS_TEMPLATE = {
    np.half: [np.random.randn(m, k).astype(np.half),
              np.random.randn(k, n).astype(np.half),
              np.zeros((m, n), dtype=np.half),
              np.int32(m), np.int32(k), np.int32(n)],
    np.single: [np.random.randn(m, k).astype(np.single),
                np.random.randn(k, n).astype(np.single),
                np.zeros((m, n), dtype=np.single),
                np.int32(m), np.int32(k), np.int32(n)],
    np.double: [np.random.randn(m, k).astype(np.double),
                np.random.randn(k, n).astype(np.double),
                np.zeros((m, n), dtype=np.double),
                np.int32(m), np.int32(k), np.int32(n)]
}

# Load kernel code
with open("gemm.cu", "r") as f:
    kernel_string = f.read()


# Theoretical performance constants (A100-specific)
PEAK_TFLOPS_FP32 = 19.5  # 19.5 TFLOPS for FP32
PEAK_TFLOPS_FP16 = 312   # 312 TFLOPS for FP16 (sparse not considered)
TDP_WATTS = 400          # Max TDP for A100

def tune_gemm(precision):
    args = ARGS_TEMPLATE[precision]

    a = np.random.randn(m, k).astype(precision)
    b = np.random.randn(k, n).astype(precision)
    c = np.zeros((m, n), dtype=precision)
    
    args = [a, b, c, np.int32(m), np.int32(k), np.int32(n)]

    # Compute theoretical minimum time
    total_flops = 2 * m * n * k
    peak_tflops = PEAK_TFLOPS_FP32 if precision == np.single else PEAK_TFLOPS_FP16
    theoretical_min_time = total_flops / (peak_tflops * 1e12)
    
    # Define tuning parameters for RQ2
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    tune_params["block_size_y"] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    tune_params["tile_size_x"] = [2, 4, 8, 16, 32]
    tune_params["tile_size_y"] = [2, 4, 8, 16, 32]
    tune_params["nvml_pwr_limit"] = get_nvml_pwr_limits(device=0, quiet=False)["nvml_pwr_limit"]


    # Define metrics with EDP + latency penalty
    metrics = OrderedDict()
    alpha, beta = 1.0, 1e6  # Penalty hyperparameters
    
    metrics["edp"] = lambda r: r["time"] * r["nvml_energy"]
    metrics["throughput_penalty"] = lambda r: beta if r["time"] > 1.05*theoretical_min_time else 0
    metrics["objective"] = lambda r: alpha*r["edp"] + r["throughput_penalty"]

    

    # Configure observers
    nsight_observer = NCUObserver(metrics=[
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__bytes.sum.per_second"
    ])
    nvml_observer = NVMLObserver(["nvml_energy", "nvml_power"])

    type_map = {
        np.half: ("half", "__half"),
        np.single: ("float", "float"),
        np.double: ("double", "double")
    }
    
    cuda_type, kernel_type = type_map[precision]
    
    # searchspace = Searchspace(tune_params, restrictions)
    # valid_configs = len(searchspace.list)
    # print(f"Valid configurations: {valid_configs}")

    # Bayesian optimization tuning
    results, env = kt.tune_kernel(
        f"tuned_matmul<{kernel_type}>",  kernel_string,
        problem_size, args, tune_params,
        compiler_options=[
            "-Xptxas=-v",
            "--expt-relaxed-constexpr", 
            "-std=c++17",
            "-Xcompiler",
            "-Wall",
            "-lcublas",
        ],
        restrictions=restrictions,
        metrics=metrics,
        strategy="bayes_opt",
        strategy_options={
            "max_fevals": 100,
            "acquisition_type": "EI"  # Expected Improvement
        },
        objective="objective",
        objective_higher_is_better=False,
        observers=[nsight_observer, nvml_observer],
        cache=f"bayes_opt_{precision.__name__}.json",
        verbose=True
    )
    for config in results:
        config['tune_params'] = tune_params
    return results, tune_params


if __name__ == "__main__":
    for precision in [np.half, np.single, np.double]:
        print(f"\n=== Tuning {precision.__name__} precision ===")
        results, tune_params = tune_gemm(precision)  # Now correctly unpacked
        top_configs = sorted(results, key=lambda x: x['objective'])[:10]
