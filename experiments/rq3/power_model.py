#!/usr/bin/env python3
import math
from gpu_common import GPUArchitecture, KernelAnalysis

class HongKimPowerEstimator:
    """
    Updated Hong–Kim GPU power estimator for modern architectures.
    All power-related constants are now read from calibration.json.
    Make sure to run calibration.py on your GPU to update these values.
    """
    def __init__(self, arch: GPUArchitecture, analysis: KernelAnalysis):
        self.arch = arch
        self.analysis = analysis
        cdata = arch.calibration_data

        # Required calibration keys – no hardcoded fallback
        required_keys = [
            "idle_power",
            "max_power_fp",
            "max_power_int",
            "max_power_sfu",
            "max_power_alu",
            "max_power_fds",
            "max_power_reg",
            "max_power_shm",
            "max_power_mem",
            "power_log_alpha",
            "power_log_beta",
            "issue_cycles",
            "short_kernel_scale",
            "max_power_total"
        ]
        for key in required_keys:
            if key not in cdata:
                raise KeyError(f"Calibration key '{key}' missing in {arch.calibration_file}. "
                               f"Please run calibration.py.")

        # Basic powers
        self.idle_power       = float(cdata["idle_power"])
        self.max_power_fp     = float(cdata["max_power_fp"])
        self.max_power_int    = float(cdata["max_power_int"])
        self.max_power_sfu    = float(cdata["max_power_sfu"])
        self.max_power_alu    = float(cdata["max_power_alu"])
        self.max_power_fds    = float(cdata["max_power_fds"])
        self.max_power_reg    = float(cdata["max_power_reg"])
        self.max_power_shmem  = float(cdata["max_power_shm"])
        self.max_power_mem    = float(cdata["max_power_mem"])
        self.short_kernel_scale = float(cdata["short_kernel_scale"])

        # concurrency
        self.log_alpha = float(cdata["power_log_alpha"])
        self.log_beta  = float(cdata["power_log_beta"])

        # Issue cycles used for accessRate
        self.issue_cycles = float(cdata["issue_cycles"])
        self.max_power_total = float(cdata["max_power_total"])
        self.min_clamped = self.idle_power  # can't go below idle

    def estimate_power(self,
                   exec_cycles: float,
                   warps_per_sm: float,
                   active_sms: int,
                   clock_rate_hz: float) -> float:
        """
        Updated power estimation that accounts for block shape in concurrency/power
        even if the total threads per block is the same.
        :param exec_cycles: total cycles from the time model
        :param warps_per_sm: concurrency from time model
        :param active_sms: number of SMs actually used
        :param clock_rate_hz: GPU clock
        :return: predicted average power in watts
        """

        if exec_cycles < 1.0:
            exec_cycles = 1.0

        def access_rate(num_insts: float) -> float:
            """
            Hong–Kim style: 
            AccessRate = (num_insts * warps_per_sm) / (exec_cycles / self.issue_cycles)
            """
            return (num_insts * warps_per_sm) / (exec_cycles / self.issue_cycles)

        # Pull out dynamic instruction counts
        fp_insts  = float(self.analysis.fp_insts)
        int_insts = float(self.analysis.int_insts)
        sfu_insts = float(self.analysis.sfu_insts)
        alu_insts = float(self.analysis.alu_insts)
        total_insts = float(self.analysis.total_insts)

        # Memory instructions
        mem_insts = float(
            self.analysis.mem_coal
            + self.analysis.mem_uncoal
            + self.analysis.mem_partial
            + self.analysis.local_insts
            + self.analysis.shared_insts
        )

        # Compute access rates for each sub-component
        AR_fp  = access_rate(fp_insts)
        AR_int = access_rate(int_insts)
        AR_sfu = access_rate(sfu_insts)
        AR_alu = access_rate(alu_insts)

        AR_fds = access_rate(total_insts)  # e.g. fetch/decode
        AR_reg = AR_fds                   # simplistic assumption for register file
        AR_shm = access_rate(self.analysis.shared_insts)
        AR_mem = access_rate(mem_insts)

        # Convert to power contributions (microbenchmark-based).
        rp_fp  = self.max_power_fp   * AR_fp
        rp_int = self.max_power_int  * AR_int
        rp_sfu = self.max_power_sfu  * AR_sfu
        rp_alu = self.max_power_alu  * AR_alu
        rp_fds = self.max_power_fds  * AR_fds
        rp_reg = self.max_power_reg  * AR_reg
        rp_shm = self.max_power_shmem* AR_shm

        # Sum SM subcomponents
        rp_sm_sub = (rp_fp + rp_int + rp_sfu + rp_alu 
                    + rp_fds + rp_reg + rp_shm + 0.0)  # no big 'const_sm_power' lump

        # Memory portion
        rp_mem = self.max_power_mem * AR_mem

        # Now incorporate block shape (X vs. Y) in memory coalescing penalty
        block_dim_x = self.analysis.block_x
        block_dim_y = self.analysis.block_y
        warp_size   = self.arch.attrs.get('WARP_SIZE', 32)

        # Evaluate coalescing “penalty” based on block_dim_x
        # e.g. if block_dim_x < warp_size, less coalescing
        # or a formula for partial coalescing:
        coalesce_eff = min(1.0, float(block_dim_x) / warp_size)
        # Example: Add up to +50% overhead if shape is suboptimal
        # rp_mem *= (1.0 + (1.0 - coalesce_eff) * 1.5)

        penalty = 1.0 + (1.0 - coalesce_eff) * self.arch.calibration_data["partial_coalesce_slope"]
        rp_mem *= penalty

        # concurrency factor from alpha,beta
        factor = math.log10(self.log_alpha * active_sms + self.log_beta)
        # test
        # factor = 1.0 + (active_sms / self.arch.sm_count) * (self.log_beta - 1.0)
        if factor < 0.0:
            factor = 0.0

        # Multiply subcomponents by concurrency factor
        # across all SMs
        total_dynamic = (self.arch.sm_count * rp_sm_sub + rp_mem) * factor

        # **Shape-based concurrency**: If block shape is tall vs. wide, 
        # we can incorporate an additional shape factor.
        # For example, if block_dim_x != block_dim_y, 
        # we assume it affects concurrency differently than a square shape.

        threads_per_block = block_dim_x * block_dim_y
        aspect_ratio = float(block_dim_x) / float(block_dim_y) if block_dim_y>0 else float(block_dim_x)
        # For example, if aspect_ratio >> 1 or << 1, concurrency may degrade.
        # We'll do a mild log-based shape factor:
        shape_factor = 1.0 + 0.1*abs(math.log(max(aspect_ratio, 1e-6)))
        # Then modulate by "compute intensity" so pure compute might see less shape impact:
        total_compute = (self.analysis.fp_insts + self.analysis.int_insts
                        + self.analysis.sfu_insts + self.analysis.alu_insts)
        total_memory  = (self.analysis.mem_coal + self.analysis.mem_uncoal
                        + self.analysis.mem_partial + self.analysis.local_insts
                        + self.analysis.shared_insts)
        compute_intensity = float(total_compute)/max(total_memory, 1.0)

        # reduce shape penalty if kernel is compute-bound
        shape_factor_adj = 1.0 + (shape_factor - 1.0)/(1.0 + compute_intensity)
        shape_factor_adj = max(1.0, min(shape_factor_adj, 1.5))

        total_dynamic *= shape_factor_adj

        # Combine with idle
        predicted = self.idle_power + total_dynamic

        # clamp
        if predicted < self.min_clamped:
            predicted = self.min_clamped
        if predicted > self.max_power_total:
            predicted = self.max_power_total
        return predicted

