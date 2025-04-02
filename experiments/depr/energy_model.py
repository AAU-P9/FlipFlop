#!/usr/bin/env python3
import math
import os
import json
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

# Local imports
from gpu_common import GPUArchitecture, compile_kernel
from time_model import HongKimExecutionTimeModel
from power_model import HongKimPowerEstimator
from PTXAnalyzer import PTXAnalyzer

def prepare_kernel_args(config, arch):
    """
    Prepare kernel arguments from the JSON 'config'.
    Returns a list of kernel arguments (device pointers or scalar values).
    """
    dims = config.get("dimensions", {})
    try:
        resolved_dims = {key: int(value) for key, value in dims.items()}
    except Exception as e:
        raise ValueError(f"Error converting 'dimensions' to integers: {e}")
    
    args_list = []
    for arg in config["kernel_args"]:
        arg_type = arg["type"]
        if arg_type == "array":
            shape = []
            for dim in arg["shape"]:
                if isinstance(dim, str):
                    if dim in resolved_dims:
                        shape.append(resolved_dims[dim])
                    else:
                        raise ValueError(f"Dimension '{dim}' not found in config: {dims}")
                else:
                    shape.append(dim)
            # Create host array
            if arg["init"] == "random":
                host_array = np.random.randn(*shape).astype(np.float32)
            elif arg["init"] == "zeros":
                host_array = np.zeros(shape, dtype=np.float32)
            else:
                raise ValueError(f"Unknown init type: {arg['init']}")
            # Allocate and copy to device
            dptr = cuda.mem_alloc(host_array.nbytes)
            cuda.memcpy_htod(dptr, host_array)
            args_list.append(dptr)

        elif arg_type == "int":
            val = arg["value"]
            if isinstance(val, str):
                # possibly a dimension name
                if val in resolved_dims:
                    val = resolved_dims[val]
                else:
                    raise ValueError(f"Dimension '{val}' not provided in config dims: {dims}")
            args_list.append(np.int32(val))

        elif arg_type == "float":
            val = arg["value"]
            if isinstance(val, str):
                try:
                    val = float(val)
                except Exception as e:
                    raise ValueError(f"Error converting float '{val}': {e}")
            args_list.append(np.float32(val))

        else:
            raise ValueError(f"Unsupported argument type: {arg_type}")
    return args_list


def main():
    """
    This script orchestrates the entire pipeline:
      1) Compile a CUDA kernel or PTX code.
      2) Parse the resulting PTX with PTXAnalyzer for instruction mix and memory patterns.
      3) Use HongKimExecutionTimeModel and HongKimPowerEstimator to estimate runtime & power.
      4) Optionally run the kernel to measure real time & power (NVML).
      5) Compare predictions vs. actual results.
    """
    import argparse
    parser = argparse.ArgumentParser(description="GPU Energy Model for any CUDA Kernel")
    parser.add_argument("kernel", help="Path to .cu source or PTX for the CUDA kernel")
    parser.add_argument("gdimx", type=int, help="Grid dimension X")
    parser.add_argument("gdimy", type=int, help="Grid dimension Y")
    parser.add_argument("bdimx", type=int, help="Block dimension X")
    parser.add_argument("bdimy", type=int, help="Block dimension Y")
    parser.add_argument("size",  type=int, help="Data size or dimension to build from")
    parser.add_argument("--runs", type=int, default=30, help="Number of repeats for measuring kernel performance")
    parser.add_argument("--config", type=str, default="", help="Path to kernel config JSON for complex argument setups")
    parser.add_argument("--calib",  type=str, default="calibration/calibration.json",
                        help="Path to your calibration JSON, e.g. from calibration.py")
    args = parser.parse_args()

    # 1) Initialize GPU architecture with your calibration
    arch = GPUArchitecture(calibration_file=args.calib)

    # 2) Compile the kernel (CU -> PTX or directly PTX)
    mod, ptx_str, ptxas_log, kname = compile_kernel(args.kernel, arch)

    # 3) Build or load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # fallback: a naive matrix multiply config
        Nside = int(math.sqrt(args.size))
        config = {
            "kernel_name": "dummyKernel",
            "dimensions": {
                "M": Nside,
                "N": Nside,
                "K": Nside
            },
            "kernel_args": [
                {"name": "A", "type": "array","dtype":"float32","shape":["M","N"],"init":"random"},
                {"name": "B", "type": "array","dtype":"float32","shape":["N","K"],"init":"random"},
                {"name": "C", "type": "array","dtype":"float32","shape":["M","K"],"init":"zeros"},
                {"name": "M", "type": "int","value":"M"},
                {"name": "N", "type": "int","value":"N"},
                {"name": "K", "type": "int","value":"K"}
            ]
        }

    # 4) Prepare kernel args
    kernel_args = prepare_kernel_args(config, arch)

    # 5) Run PTXAnalyzer to parse instructions
    analyzer = PTXAnalyzer(
        ptx_code=ptx_str,
        ptxas_log=ptxas_log,
        arch=arch,  # so it knows clock rate, warp size, etc. if needed
        block_x=args.bdimx,
        block_y=args.bdimy,
        config=config
    )
    analysis = analyzer.analyze()  # returns a KernelAnalysis structure with instruction counts, etc.

    # 6) Estimate time using the time model
    from time_model import HongKimExecutionTimeModel
    time_model = HongKimExecutionTimeModel(arch, analysis, (args.gdimx,args.gdimy), (args.bdimx,args.bdimy))
    est_time_ns = time_model.estimate_time_ns()
    exec_cycles = (est_time_ns * arch.clock_rate_hz)/1e9

    # concurrency
    warp_size = arch.attrs.get("WARP_SIZE",32)
    threads_per_block = args.bdimx*args.bdimy
    warps_per_block   = (threads_per_block + warp_size -1)//warp_size
    blocks_per_sm     = time_model._calc_blocks_per_sm(threads_per_block)
    warps_per_sm      = float(blocks_per_sm*warps_per_block)
    active_sms        = arch.sm_count

    # 7) Estimate power
    from power_model import HongKimPowerEstimator
    power_estimator = HongKimPowerEstimator(arch, analysis)
    predicted_power = power_estimator.estimate_power(
        exec_cycles,
        warps_per_sm,
        active_sms,
        arch.clock_rate_hz
    )

    # 8) Run kernel & measure actual time and power
    kernel_func = mod.get_function(kname)
    

    actual_ns = float(actual_us)*1e3
    diff_pct = abs(est_time_ns - actual_ns)/max(actual_ns,1.0)*100.0

    print(f"[RESULT] Kernel={kname}")
    print(f"Analysis = {analysis}")
    print(f"Estimated Time (ns) = {est_time_ns:.2f}")
    # print(f"Actual Time   (ns) = {actual_ns:.2f}")
    # print(f"-> Diff (%) = {diff_pct:.2f}")
    print(f"WarpsPerSM = {warps_per_sm:.2f}")
    print(f"Predicted Power (W) = {predicted_power:.2f}")
    # print(f"Actual Power (W) = {actual_pw:.2f}")

    # -- Additional debug prints for static analysis results--
    print(f"MemCoal={analysis.mem_coal}")
    print(f"MemUncoal={analysis.mem_uncoal}")
    print(f"MemPartial={analysis.mem_partial}")
    print(f"LocalInsts={analysis.local_insts}")
    print(f"SharedInsts={analysis.shared_insts}")
    print(f"SynchInsts={analysis.synch_insts}")
    print(f"FpInsts={analysis.fp_insts}")
    print(f"IntInsts={analysis.int_insts}")
    print(f"SfuInsts={analysis.sfu_insts}")
    print(f"AluInsts={analysis.alu_insts}")
    print(f"TotalInsts={analysis.total_insts}")
    print(f"RegsPerThread={analysis.registers_per_thread}")
    print(f"SharedMemBytes={analysis.shared_mem_bytes}")

    # print grid and block shapes
    print(f"GridX={args.gdimx}")
    print(f"GridY={args.gdimy}")
    print(f"BlockX={args.bdimx}")
    print(f"BlockY={args.bdimy}")


if __name__ == "__main__":
    main()
