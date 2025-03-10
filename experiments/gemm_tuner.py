import numpy as np
from collections import OrderedDict
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.ncu import NCUObserver
import kernel_tuner as kt
from kernel_tuner.observers.nvml import get_nvml_pwr_limits, get_nvml_gr_clocks, get_nvml_mem_clocks, get_idle_power

def analyze_efficiency(config):
    with nvtx.annotate("kernel_profiling"):
        result = run_kernel(config, counters)
        
    intensity = (2*m*n*k) / (a.nbytes + b.nbytes + c.nbytes)
    bound_type = "compute" if intensity > 100 else "memory"
    
    print(f"Kernel is {bound_type}-bound with:")
    print(f" - SM Utilization: {result.counters.sm_util}%")
    print(f" - L1 Cache Hit Rate: {result.cache_hit_rate}%")
    print(f" - DRAM Throughput: {result.dram_throughput} GB/s")


def tune_gemm(precision):
    # m, n, k = 4096, 4096, 4096
    m = np.int32(4096)
    n = np.int32(4096)
    k = np.int32(4096)
    
    a = np.random.randn(m, k).astype(precision)
    b = np.random.randn(k, n).astype(precision)
    c = np.zeros((m, n), dtype=precision)
    
    args = [a, b, c, m, k, n]
    problem_size = (n, m)

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tune_params["block_size_y"] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tune_params["tile_size_x"] = [1, 2, 4]
    tune_params["tile_size_y"] = [1, 2, 4]
    tune_params["nvml_pwr_limit"] = get_nvml_pwr_limits(device=0, n=5,quiet=False)["nvml_pwr_limit"]

    # tune_params["vector_type"] = [("float", 1), ("float2", 2), ("float4", 4)]  # (type, width)
    # tune_params["loop_unroll_factor"] = [0, 2, 4]
    # tune_params["PREFETCH_INPUT"] = [0, 1]
    # tune_params["my_type"] = ["__half" , "float", "double"]


    # restrictions = [
    #     lambda p: (p["block_size_x"] * p["block_size_y"]) <= 1024,
    #     lambda p: (p["tile_size_x"] % p["vector_type"][1]) == 0,
    #     lambda p: p["tile_size_y"] <= p["block_size_y"]
    # ]

    restrictions = [
        "(block_size_x * block_size_y) <= 1024",       # SM thread capacity
        "(block_size_x * block_size_y) % 32 == 0",        # Warp size
        # "(tile_size_x % vector_type[1]) == 0",          # Vector alignment
        "tile_size_y <= block_size_y"                 # Tiling validity
    ]

    counters = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",  # Direct occupancy percentage
        "sm__maximum_warps_per_active_cycle_pct",  # Theoretical max occupancy
        # "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
        # "lts__t_sectors.sum",
        # "dram__sectors.sum"
        # "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed", 
        # "dram__bytes.sum.per_second",
        # "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        # "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
    ]

    metrics = OrderedDict()
    metrics["edp"] = lambda r: r["time"] * r["nvml_energy"]
    metrics["flops_watt"] = lambda r: (2*m*n*k) / (r["nvml_energy"] * 1e3)
    metrics["occupancy"] = lambda r: r.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0)
    # metrics["BankConflict"] = lambda p: (p["l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"] / 
    #                           p["l1tex__data_pipe_lsu_wavefronts.sum"]) * 100

    with open("gemm.cu", "r") as f:
        kernel_string = f.read()

    type_map = {
        np.half: ("half", "__half"),
        np.single: ("float", "float"),
        np.double: ("double", "double")
    }
    
    cuda_type, kernel_type = type_map[precision]
    
    nsight_observer = NCUObserver(metrics=counters)
    nvml_observer = NVMLObserver(
        ["nvml_energy", "nvml_power"]
    )

    result = kt.tune_kernel(
        f"tuned_matmul<{kernel_type}>", kernel_string,
        problem_size, args, tune_params,
        # lang="cupy",
        compiler_options=[
            "-Xptxas=-v",
            "--expt-relaxed-constexpr",
            # "-D__CUDA_NO_HALF_OPERATORS__",
            "-std=c++17",
            "-Xcompiler",
            "-Wall",
            "-lcublas",
        ],
        restrictions=restrictions,
        metrics=metrics,
        # strategy="bayesian",
        # strategy_options={"max_fevals": 200},
        observers=[nsight_observer, nvml_observer],
        verbose=True,
        cache=f"power_tuning_{precision.__name__}.json"
    )
    
    return result

def analyze_results(results):
    for power, config in results:
        print(f"\nPower Limit: {power}W")
        print(f"Best EDP: {config['edp']:.2f} J·s")
        print(f"Performance: {2*m*n*k/config['time']/1e12:.2f} TFLOPS")
        print(f"Occupancy: {config['occupancy']*100:.1f}%")
        print("Optimal Parameters:")
        for k, v in config.items():
            if k.startswith("_"): continue
            print(f"  {k:15}: {v}")

if __name__ == "__main__":
    precisions = [np.half, np.single, np.double]
    
    for precision in precisions:
        print(f"\nTuning {precision.__name__} precision:")
        results = tune_gemm(precision)
        # analyze_results(results)
