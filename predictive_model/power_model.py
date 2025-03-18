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

        required_keys = [
            "idle_power",
            "max_power_fp",
            "max_power_int",
            "max_power_sfu",
            "max_power_alu",
            "max_power_fds",
            "max_power_reg",
            "max_power_shm",
            "const_sm_power",
            "max_power_mem",
            "power_log_alpha",
            "power_log_beta",
            "max_power_total",
            "issue_cycles"
        ]
        for key in required_keys:
            if key not in cdata:
                raise KeyError(f"Calibration key '{key}' missing. Please run calibration.py.")

        self.idle_power      = float(cdata["idle_power"])
        self.max_power_fp    = float(cdata["max_power_fp"])
        self.max_power_int   = float(cdata["max_power_int"])
        self.max_power_sfu   = float(cdata["max_power_sfu"])
        self.max_power_alu   = float(cdata["max_power_alu"])
        self.max_power_fds   = float(cdata["max_power_fds"])
        self.max_power_reg   = float(cdata["max_power_reg"])
        self.max_power_shmem = float(cdata["max_power_shm"])
        self.const_sm_power  = float(cdata["const_sm_power"])
        self.max_power_mem   = float(cdata["max_power_mem"])

        self.log_alpha       = float(cdata["power_log_alpha"])
        self.log_beta        = float(cdata["power_log_beta"])
        self.min_clamped     = self.idle_power  # Minimum power = idle power.
        self.max_clamped     = float(cdata["max_power_total"])

        self.issue_cycles = float(cdata["issue_cycles"])

    def estimate_power(self,
                       exec_cycles: float,
                       warps_per_sm: float,
                       active_sms: int,
                       clock_rate_hz: float) -> float:
        if exec_cycles < 1.0:
            exec_cycles = 1.0

        # Compute access rates based on the effective cycles per instruction.
        def access_rate(num_insts: float) -> float:
            return (num_insts * warps_per_sm) / (exec_cycles / self.issue_cycles)

        fp_insts  = float(self.analysis.fp_insts)
        int_insts = float(self.analysis.int_insts)
        sfu_insts = float(self.analysis.sfu_insts)
        alu_insts = float(self.analysis.alu_insts)
        total_comp = fp_insts + int_insts + sfu_insts + alu_insts
        total_insts = float(self.analysis.total_insts)
        mem_insts = float(self.analysis.mem_coal + self.analysis.mem_uncoal +
                          self.analysis.mem_partial + self.analysis.local_insts +
                          self.analysis.shared_insts)

        AR_fp  = access_rate(fp_insts)
        AR_int = access_rate(int_insts)
        AR_sfu = access_rate(sfu_insts)
        AR_alu = access_rate(alu_insts)
        AR_fds = access_rate(total_insts)
        AR_reg = access_rate(total_insts)
        AR_shm = access_rate(float(self.analysis.shared_insts))
        AR_mem = access_rate(mem_insts)

        rp_fp   = self.max_power_fp   * AR_fp
        rp_int  = self.max_power_int  * AR_int
        rp_sfu  = self.max_power_sfu  * AR_sfu
        rp_alu  = self.max_power_alu  * AR_alu
        rp_fds  = self.max_power_fds  * AR_fds
        rp_reg  = self.max_power_reg  * AR_reg
        rp_shm  = self.max_power_shmem* AR_shm

        # Sum all SM dynamic power subcomponents plus the constant SM overhead.
        rp_sm_sub = (rp_fp + rp_int + rp_sfu + rp_alu + rp_fds + rp_reg + rp_shm + self.const_sm_power)
        rp_mem = self.max_power_mem * AR_mem

        bx = self.analysis.block_x
        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        coalesce_eff = min(1.0, (bx * 1.0)/warp_size)
        rp_mem *= (1.0 + (1.0 - coalesce_eff)*1.5)  # +50% penalty

        # Total power consumed by all active SMs.
        Max_SM = self.arch.sm_count * rp_sm_sub
        factor = math.log10(self.log_alpha * active_sms + self.log_beta)
        factor = max(factor, 0.1)
        runtime_power = (Max_SM + rp_mem) * factor


        # Apply block shape-dependent correction
        threads_per_block = self.analysis.block_x * self.analysis.block_y
        block_dim_x, block_dim_y = self.analysis.block_x, self.analysis.block_y
        
        # Coalescing efficiency based on blockDim.x
        ce_x = min(1.0, block_dim_x / warp_size)
        
        # Shape balance factor
        aspect_ratio = block_dim_x / block_dim_y if block_dim_y > 0 else 1.0
        shape_balance = 1.0 + 0.1 * abs(math.log(max(aspect_ratio, 1e-6)))
        
        # Power adjustment factor: increases power for less efficient shapes
        power_factor = shape_balance / ce_x if ce_x > 0 else shape_balance
        power_factor = max(1.0, min(power_factor, 1.5))  # Clamp between 1.0 and 1.5
        runtime_power *= power_factor

        predicted = runtime_power + self.idle_power

        if predicted < self.min_clamped:
            predicted = self.min_clamped
        elif predicted > self.max_clamped:
            predicted = self.max_clamped
        return predicted
