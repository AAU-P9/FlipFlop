#!/usr/bin/env python3
import math
from gpu_common import GPUArchitecture, KernelAnalysis

class HongKimExecutionTimeModel:
    """
    Updated Hong–Kim GPU execution time model for modern architectures.
    All parameters (latencies, departure delays, issue cycles, etc.) are now obtained
    from the calibration.json file (via GPUArchitecture.calibration_data).
    Run calibration.py on your target GPU to update calibration.json before use.
    """
    def __init__(self,
                 arch: GPUArchitecture,
                 analysis: KernelAnalysis,
                 grid_xy,
                 block_xy):
        self.arch = arch
        self.analysis = analysis
        self.grid_xy  = grid_xy   # (gdimx, gdimy)
        self.block_xy = block_xy  # (bdimx, bdimy)

        cdata = arch.calibration_data
        # Ensure all required calibration keys are present (no hardcoded fallbacks)
        required_keys = [
            "Mem_LD_coal_ns",
            "Mem_LD_uncoal_ns",
            "Mem_LD_partial_ns",
            "Mem_LD_shared_ns",
            "Mem_LD_local_ns",
            "issue_cycles",
            "baseline_kernel_overhead_ns",
            "Departure_del_coal_s",
            "Departure_del_uncoal_s"
        ]
        for key in required_keys:
            if key not in cdata:
                raise KeyError(f"Calibration key '{key}' missing. Please run calibration.py on your device.")

        self.Mem_coal_ns    = float(cdata["Mem_LD_coal_ns"])
        self.Mem_uncoal_ns  = float(cdata["Mem_LD_uncoal_ns"])
        self.Mem_partial_ns = float(cdata["Mem_LD_partial_ns"])
        self.Mem_shared_ns  = float(cdata["Mem_LD_shared_ns"])
        self.Mem_local_ns   = float(cdata["Mem_LD_local_ns"])
        self.issue_cycles   = float(cdata["issue_cycles"])
        self.baseline_ns    = float(cdata["baseline_kernel_overhead_ns"])

        self.Dep_coal_s    = float(cdata["Departure_del_coal_s"])
        self.Dep_uncoal_s  = float(cdata["Departure_del_uncoal_s"])
        # Use the average for partial accesses.
        self.Dep_part_s    = 0.5 * (self.Dep_coal_s + self.Dep_uncoal_s)
        # For shared/local departures, use measured values if available; otherwise, derive from clock rate.
        self.Dep_shared_s  = float(cdata.get("Departure_del_shared_s", 1.0/self.arch.clock_rate_hz))
        self.Dep_local_s   = float(cdata.get("Departure_del_local_s", self.Dep_uncoal_s))

    def estimate_time_ns(self) -> float:
        # Calculate total blocks and warps.
        gx, gy = self.grid_xy
        bx, by = self.block_xy

        print(f"Grid: {gx}x{gy}, Block: {bx}x{by}")

        total_blocks = gx * gy
        threads_per_block = bx * by
        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        warps_per_block = (threads_per_block + warp_size - 1) // warp_size
        total_warps = total_blocks * warps_per_block

        # Calculate memory instruction counts (global, local, shared).
        mem_coal   = float(self.analysis.mem_coal * total_warps)
        mem_uncoal = float(self.analysis.mem_uncoal * total_warps)
        mem_part   = float(self.analysis.mem_partial * total_warps)
        mem_local  = float(self.analysis.local_insts * total_warps)
        mem_shr    = float(self.analysis.shared_insts * total_warps)
        global_count = mem_coal + mem_uncoal + mem_part

        # Compute instruction counts.
        comp_fp  = float(self.analysis.fp_insts * total_warps)
        comp_int = float(self.analysis.int_insts * total_warps)
        comp_sfu = float(self.analysis.sfu_insts * total_warps)
        comp_alu = float(self.analysis.alu_insts * total_warps)
        comp_sum = comp_fp + comp_int + comp_sfu + comp_alu
        sync_count = float(self.analysis.synch_insts * total_warps)

        # Convert latencies from ns to seconds.
        lat_coal   = self.Mem_coal_ns   * 1e-9
        lat_uncoal = self.Mem_uncoal_ns * 1e-9
        lat_part   = self.Mem_partial_ns* 1e-9
        lat_shared = self.Mem_shared_ns * 1e-9
        lat_local  = self.Mem_local_ns  * 1e-9

        # Compute weighted average global latency.
        if global_count > 1e-9:
            global_lat = (mem_coal * lat_coal + mem_uncoal * lat_uncoal + mem_part * lat_part) / global_count
        else:
            global_lat = 0.0

        mem_total = global_count + mem_local + mem_shr
        if mem_total < 1:
            comp_cycles = comp_sum * self.issue_cycles
            return (comp_cycles / self.arch.clock_rate_hz) * 1e9 + self.baseline_ns

        frac_global = global_count / mem_total
        frac_local  = mem_local / mem_total
        frac_shared = mem_shr / mem_total

        coalesce_eff = min(1.0, (bx * 1.0)/warp_size)
        effective_global_lat = global_lat * (1.0 + (1.0 - coalesce_eff)*2.0)

        avg_mem_lat = (effective_global_lat * frac_global + lat_local * frac_local + lat_shared * frac_shared)

        # Estimate cycles.
        mem_cycles  = mem_total * (avg_mem_lat * self.arch.clock_rate_hz)
        comp_cycles = comp_sum * self.issue_cycles

        if global_count > 1e-9:
            w_coal = mem_coal / global_count
            w_un   = mem_uncoal / global_count
            w_part = mem_part / global_count
            dd_global = w_coal * self.Dep_coal_s + w_un * self.Dep_uncoal_s + w_part * self.Dep_part_s
        else:
            dd_global = 0.0

        dd_local = self.Dep_local_s
        dd_shared = self.Dep_shared_s
        mem_dep = (dd_global * frac_global + dd_local * frac_local + dd_shared * frac_shared)
        MWP_woBW_full = avg_mem_lat / mem_dep if mem_dep > 1e-15 else 1e6

        # Determine active warps per SM.
        blocks_per_sm = self._calc_blocks_per_sm(threads_per_block)
        warps_per_sm = blocks_per_sm * warps_per_block
        N = float(warps_per_sm)
        MWP_woBW = min(MWP_woBW_full, N)

        memBW_Bps = self.arch.memory_bandwidth_gbps() * 1e9
        load_bytes_warp = 4.0 * warp_size
        warp_bw = (load_bytes_warp / avg_mem_lat) if avg_mem_lat > 1e-15 else 1e12
        sm_bw = memBW_Bps / max(self.arch.sm_count, 1)
        MWP_peak_BW = sm_bw / warp_bw if warp_bw > 1e-9 else 1e6
        MWP = min(MWP_woBW, MWP_peak_BW, N)

        CWP_full = (mem_cycles + comp_cycles) / comp_cycles if comp_cycles > 1e-9 else N
        CWP = min(CWP_full, N)
        comp_p = comp_cycles / mem_total if mem_total > 0 else 0.0
        Mem_cy = mem_cycles
        Comp_cy = comp_cycles
        reps = self._calc_block_reps(total_blocks, blocks_per_sm)

        # Choose execution cycle formula based on relative MWP and CWP.
        if abs(MWP - N) < 1e-3 and abs(CWP - N) < 1e-3:
            totalCycles = (Mem_cy + Comp_cy + comp_p * (MWP - 1)) * reps
        elif (CWP >= MWP) or (Comp_cy > Mem_cy):
            totalCycles = (Mem_cy * (N / MWP) + comp_p * (MWP - 1)) * reps
        else:
            Mem_L_cycles = avg_mem_lat * self.arch.clock_rate_hz
            totalCycles = (Mem_L_cycles + Comp_cy * N) * reps

        if mem_dep > 1e-15 and warps_per_sm > 1:
            depCycles = mem_dep * self.arch.clock_rate_hz
            blocks_psm = blocks_per_sm
            # Adjust synchronization cost to use the dynamic sync_count.
            syncCycles = depCycles * (MWP - 1) * sync_count / total_blocks * blocks_psm * reps
            totalCycles += syncCycles

        # Apply block shape-dependent correction
        block_dim_x, block_dim_y = self.block_xy[0], self.block_xy[1]
        
        # Coalescing efficiency
        ce_x = min(1.0, block_dim_x / warp_size)
        
        # Shape balance factor
        aspect_ratio = block_dim_x / block_dim_y if block_dim_y > 0 else 1.0
        shape_balance = 1.0 + 0.2 * abs(math.log(max(aspect_ratio, 1e-6)))
        
        # Compute intensity: ratio of compute to memory instructions
        total_compute = self.analysis.fp_insts + self.analysis.int_insts + \
                        self.analysis.sfu_insts + self.analysis.alu_insts
        total_memory = self.analysis.mem_coal + self.analysis.mem_uncoal + \
                    self.analysis.mem_partial + self.analysis.local_insts + \
                    self.analysis.shared_insts
        compute_intensity = total_compute / max(total_memory, 1.0)
        
        # Adjusted shape factor: reduce impact for compute-bound kernels
        base_factor = shape_balance / ce_x if ce_x > 0 else shape_balance
        shape_factor = 1.0 + (base_factor - 1.0) / (1.0 + compute_intensity)
        shape_factor = max(1.0, min(shape_factor, 1.5))  # Tighter clamp
        totalCycles *= shape_factor

        kernel_ns = (totalCycles / self.arch.clock_rate_hz) * 1e9 + self.baseline_ns
        return kernel_ns

    def _calc_blocks_per_sm(self, threads_per_block: int) -> int:
        arch_attrs = self.arch.attrs
        max_thr_sm = arch_attrs['MAX_THREADS_PER_MULTIPROCESSOR']
        tlim = max_thr_sm // threads_per_block if threads_per_block > 0 else 1
        max_regs_sm = arch_attrs['MAX_REGISTERS_PER_MULTIPROCESSOR']
        regs_needed = self.analysis.registers_per_thread * threads_per_block
        rlim = max_regs_sm // regs_needed if regs_needed > 0 else 1
        max_shm_sm = arch_attrs['MAX_SHARED_MEMORY_PER_MULTIPROCESSOR']
        smlim = max_shm_sm // self.analysis.shared_mem_bytes if self.analysis.shared_mem_bytes > 0 else 1
        hw_blocklim = arch_attrs.get('MAX_BLOCKS_PER_MULTIPROCESSOR', 32)
        blocks_possible = min(tlim, rlim, smlim, hw_blocklim)
        return max(blocks_possible, 1)

    def _calc_block_reps(self, total_blocks: int, blocks_per_sm: int) -> float:
        sm_count = self.arch.sm_count
        blocks_per_round = blocks_per_sm * sm_count
        import math
        return float(math.ceil(total_blocks / blocks_per_round))
