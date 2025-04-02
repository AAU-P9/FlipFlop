#!/usr/bin/env python3
import sys
import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from typing import Tuple
from gpu_common import GPUArchitecture, PTXAnalyzer, KernelAnalysis, compile_kernel, benchmark_kernel

class ExecutionTimeModel:
    def __init__(self, arch: GPUArchitecture, analysis: KernelAnalysis,
                 grid: Tuple[int,int], block: Tuple[int,int], sub_phases=None):
        self.arch = arch
        self.analysis = analysis
        self.grid = grid
        self.block = block
        self.sub_phases = sub_phases
        wsize = arch.attrs['WARP_SIZE']
        self.warp_size = wsize if wsize>0 else 32

        cdata = arch.calibration_data
        concurrency_data = cdata.get("concurrency_data", {})
        self.partial_coalescing_factor = concurrency_data.get("partial_coalesce_factor", 1.0)

        self.baseline_ns      = cdata.get("baseline_kernel_overhead_ns", 0.0)
        self.short_ns         = cdata.get("short_kernel_penalty_ns", 0.0)
        self.Mem_coal_ns      = cdata.get("Mem_LD_coal_ns", 500.0)
        self.Mem_uncoal_ns    = cdata.get("Mem_LD_uncoal_ns", 800.0)
        self.Mem_partial_ns   = cdata.get("Mem_LD_partial_ns",650.0)
        self.Mem_shared_ns    = cdata.get("Mem_LD_shared_ns", 20.0)
        self.Mem_local_ns     = cdata.get("Mem_LD_local_ns", 100.0)
        self.issue_cycles     = cdata.get("issue_cycles",4)
        self.Dep_coal_s       = cdata.get("Departure_del_coal_s", 5.0e-8)
        self.Dep_uncoal_s     = cdata.get("Departure_del_uncoal_s", 2.0e-7)
        self.coalesced_bytes  = cdata.get("coalesced_bytes",128)
        self.eff_bw_gbps      = cdata.get("effective_mem_bw_gbps", arch.memory_bandwidth_gbps)
        self.clock_rate       = arch.clock_rate_hz
        self.sm_count         = arch.sm_count
        self.concurrency_ratio_single = cdata.get("concurrency_ratio_single", 1.0)

    def estimate_time_ns(self) -> float:
        if self.sub_phases:
            total_ns = 0.0
            for (sub_analysis, frac) in self.sub_phases:
                t_sub = self._estimate_time_subkernel(sub_analysis)
                total_ns += t_sub * frac
            total_ns += self.baseline_ns
            if total_ns < 1e5:
                total_ns += self.short_ns
            return total_ns
        else:
            t_ns = self._estimate_time_subkernel(self.analysis)
            t_ns += self.baseline_ns
            if t_ns < 1e5:
                t_ns += self.short_ns
            return t_ns

    def _estimate_time_subkernel(self, analysis_obj: KernelAnalysis) -> float:
        total_insts = max(1, analysis_obj.total_insts)
        comp_cycles = total_insts * self.issue_cycles
        active_warps, reps = self._compute_active_blocks_and_reps(analysis_obj)

        n_coal = analysis_obj.mem_coal
        n_un   = analysis_obj.mem_uncoal
        n_part = analysis_obj.mem_partial
        n_loc  = analysis_obj.local_insts
        n_shr  = analysis_obj.shared_insts
        g_insts = n_coal + n_un + n_part

        lat_coal_ns   = self.Mem_coal_ns
        lat_un_ns     = self.Mem_uncoal_ns
        lat_part_ns   = self.Mem_partial_ns * self.partial_coalescing_factor
        lat_shared_ns = self.Mem_shared_ns
        lat_local_ns  = self.Mem_local_ns

        dd_coal = self.Dep_coal_s
        dd_un   = self.Dep_uncoal_s
        dd_part = 0.5*(dd_coal + dd_un)

        lat_coal_s   = lat_coal_ns * 1e-9
        lat_un_s     = lat_un_ns   * 1e-9
        lat_part_s   = lat_part_ns * 1e-9
        lat_shared_s = lat_shared_ns * 1e-9
        lat_local_s  = lat_local_ns  * 1e-9

        mem_insts = g_insts + n_loc + n_shr
        if mem_insts < 1:
            time_cycles = comp_cycles * reps
            return time_cycles / self.clock_rate * 1e9

        frac_global = g_insts / mem_insts
        frac_local  = n_loc   / mem_insts
        frac_shared = n_shr   / mem_insts

        w_coal = (n_coal / g_insts) if g_insts>0 else 0
        w_un   = (n_un   / g_insts) if g_insts>0 else 0
        w_part = (n_part / g_insts) if g_insts>0 else 0
        global_lat_s = lat_coal_s*w_coal + lat_un_s*w_un + lat_part_s*w_part

        Mem_L = global_lat_s * frac_global + lat_local_s * frac_local + lat_shared_s * frac_shared
        global_dep = dd_coal*w_coal + dd_un*w_un + dd_part*w_part
        local_dep  = 1.0/self.clock_rate
        shared_dep = 1.0/self.clock_rate
        mem_dep = global_dep*frac_global + local_dep*frac_local + shared_dep*frac_shared

        concurrency_ratio = self.concurrency_ratio_single
        if mem_dep>1e-15:
            MWP_woBW_full = Mem_L / mem_dep
        else:
            MWP_woBW_full = active_warps
        MWP_woBW = min(MWP_woBW_full, active_warps)

        bps = self.eff_bw_gbps * 1e9
        warp_bw = self.coalesced_bytes / max(global_lat_s,1e-15)
        sm_bw   = bps / max(self.sm_count,1)
        MWP_bw  = sm_bw / max(warp_bw,1e-15)
        MWP     = min(MWP_woBW, MWP_bw, active_warps)

        mem_cycles_coal   = lat_coal_s*self.clock_rate*n_coal
        mem_cycles_uncoal = lat_un_s  *self.clock_rate*n_un
        mem_cycles_part   = lat_part_s*self.clock_rate*n_part
        mem_global_cy = mem_cycles_coal + mem_cycles_uncoal + mem_cycles_part
        mem_local_cy  = lat_local_s*self.clock_rate*n_loc
        mem_shared_cy = lat_shared_s*self.clock_rate*n_shr
        mem_total_cy  = mem_global_cy + mem_local_cy + mem_shared_cy

        if comp_cycles>1:
            CWP_full = (mem_total_cy + comp_cycles)/float(comp_cycles)
        else:
            CWP_full = active_warps
        CWP = min(CWP_full, active_warps)

        N = active_warps
        if (abs(MWP - N)<1e-3) and (abs(CWP - N)<1e-3):
            comp_p = comp_cycles/float(mem_insts)
            time_per_rep_cy = (mem_total_cy + comp_cycles) + comp_p*(MWP-1)
        elif (CWP >= MWP) or (comp_cycles>mem_total_cy):
            comp_p = comp_cycles/float(mem_insts)
            time_per_rep_cy = mem_total_cy*(N/MWP) + comp_p*(MWP-1)
        else:
            time_per_rep_cy = (Mem_L*self.clock_rate) + comp_cycles*N

        synch_cost_cy = 0.0
        if analysis_obj.synch_insts>0:
            dep_delay_cy = mem_dep*self.clock_rate
            blocks_per_sm = self._calc_blocks_per_sm(analysis_obj)
            synch_cost_cy = dep_delay_cy*(MWP-1)*analysis_obj.synch_insts*blocks_per_sm*reps

        total_cycles = time_per_rep_cy*reps + synch_cost_cy
        total_ns = total_cycles/self.clock_rate*1e9
        return total_ns

    def _compute_active_blocks_and_reps(self, analysis_obj: KernelAnalysis) -> Tuple[float,float]:
        blocks_per_sm = self._calc_blocks_per_sm(analysis_obj)
        threads_per_block = self.block[0]*self.block[1]
        warps_per_block = math.ceil(threads_per_block/self.warp_size)
        active_warps = blocks_per_sm*warps_per_block
        total_blocks = float(self.grid[0]*self.grid[1])
        reps = math.ceil(total_blocks/(blocks_per_sm*self.arch.sm_count))
        return active_warps, reps

    # In time_model.py: Update _calc_blocks_per_sm
    def _calc_blocks_per_sm(self, analysis_obj: KernelAnalysis) -> int:
        block_threads = self.block[0] * self.block[1]
        
        # 1. Thread-limited occupancy
        max_threads_per_sm = self.arch.attrs['MAX_THREADS_PER_MULTIPROCESSOR']
        threads_lim = max_threads_per_sm // block_threads
        
        # 2. Shared memory-limited occupancy
        smem_per_block = analysis_obj.shared_mem_bytes
        smem_lim = (self.arch.attrs['MAX_SHARED_MEMORY_PER_MULTIPROCESSOR'] 
                    // max(smem_per_block, 1))
        
        # 3. Register-limited occupancy
        regs_per_block = analysis_obj.registers_per_thread * block_threads
        regs_lim = (self.arch.attrs['MAX_REGISTERS_PER_MULTIPROCESSOR'] 
                // max(regs_per_block, 1))
        
        # Use MINIMUM of all constraints
        blocks_possible = min(threads_lim, smem_lim, regs_lim)
        
        # GPU hardware typically limits blocks/SM to 32
        blocks_possible = min(blocks_possible, 32)  # <-- FIX: Add this line
        
        return max(blocks_possible, 1)  # Ensure at least 1 block

    def get_concurrency_info(self) -> Tuple[float, float]:
        active_warps, _ = self._compute_active_blocks_and_reps(self.analysis)
        t_ns = self.estimate_time_ns()
        sm_cycles = t_ns*(self.clock_rate/1e9)
        return float(active_warps), float(sm_cycles)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    run_parser = subparsers.add_parser('run', help="Run the time model")
    run_parser.add_argument("kernel", help="kernel file path")
    run_parser.add_argument("gdimx", type=int)
    run_parser.add_argument("gdimy", type=int)
    run_parser.add_argument("bdimx", type=int)
    run_parser.add_argument("bdimy", type=int)
    run_parser.add_argument("size", type=int)
    run_parser.add_argument("runs", type=int)

    args = parser.parse_args()
    if args.mode == 'run':
        arch = GPUArchitecture()
        if not arch.has_calibration():
            print("[WARNING] No calibration found, using fallback.")
        mod, ptx, ptxas_log, kname = compile_kernel(args.kernel, arch)
        kernel_func = mod.get_function(kname)

        dim = int(math.sqrt(args.size))
        M = dim
        N = dim
        K = dim

        np_a = np.random.randn(M*N).astype(np.float32)
        np_b = np.random.randn(N*K).astype(np.float32)
        np_c = np.zeros((M*K), dtype=np.float32)

        d_a = cuda.mem_alloc(np_a.nbytes)
        d_b = cuda.mem_alloc(np_b.nbytes)
        d_c = cuda.mem_alloc(np_c.nbytes)
        cuda.memcpy_htod(d_a, np_a)
        cuda.memcpy_htod(d_b, np_b)

        analyzer = PTXAnalyzer(ptx, ptxas_log, arch, args.bdimx, args.bdimy)
        analysis = analyzer.analyze()

        tm = ExecutionTimeModel(arch, analysis, (args.gdimx, args.gdimy), (args.bdimx, args.bdimy))
        estimated_ns = tm.estimate_time_ns()

        actual_us = benchmark_kernel(kernel_func, [d_a, d_b, d_c, np.int32(M), np.int32(N), np.int32(K)],
                                     grid=(args.gdimx, args.gdimy), block=(args.bdimx, args.bdimy),
                                     runs=args.runs)
        actual_ns = actual_us * 1e3
        diff_pct = abs(estimated_ns - actual_ns)/max(actual_ns,1e-9)*100.0

        print(f"[RESULT] Kernel={kname}")
        print(f"PTX Analysis={analysis}")
        print(f"Estimated time (ns)= {estimated_ns:.2f}")
        print(f"Actual median (ns)= {actual_ns:.2f}")
        print(f"Diff (%)= {diff_pct:.2f}")
    else:
        parser.print_help()

if __name__=='__main__':
    main()
