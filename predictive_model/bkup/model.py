import sys
import math
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule, compile
import numpy as np
from typing import Dict, Tuple
import re
from dataclasses import dataclass
import os
import json

#############################################################################
# 1) GPU ARCHITECTURE
#############################################################################

class GPUArchitecture:
    """
    Holds GPU device attributes and default/fallback parameters.
    """
    def __init__(self, device_id=0):
        self.device = cuda.Device(device_id)
        self.compute_capability = self.device.compute_capability()
        self.default_params = self._default_arch_params()
        self.attrs = self._fetch_device_attributes()

    def _fetch_device_attributes(self) -> Dict:
        da = cuda.device_attribute
        return {
            'MULTIPROCESSOR_COUNT': self.device.get_attribute(da.MULTIPROCESSOR_COUNT),
            'MEMORY_CLOCK_RATE': self.device.get_attribute(da.MEMORY_CLOCK_RATE),
            'GLOBAL_MEMORY_BUS_WIDTH': self.device.get_attribute(da.GLOBAL_MEMORY_BUS_WIDTH),
            'MAX_THREADS_PER_MULTIPROCESSOR': self.device.get_attribute(da.MAX_THREADS_PER_MULTIPROCESSOR),
            'MAX_SHARED_MEMORY_PER_MULTIPROCESSOR': self.device.get_attribute(da.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR),
            'CLOCK_RATE': self.device.get_attribute(da.CLOCK_RATE),
        }

    def _default_arch_params(self) -> Dict:
        """
        Fallback parameters (in seconds for latencies).
        """
        return {
            # For a simpler demonstration, we keep them in seconds
            'Mem_LD': 300e-9,  # base DRAM round-trip
            'Departure_del_coal': 5e-9,
            'Departure_del_uncoal': 20e-9,
            # We'll treat each warp instruction as 4 cycles, as in the paper
            'issue_cycles': 4
        }

    @property
    def sm_count(self) -> int:
        return self.attrs['MULTIPROCESSOR_COUNT']

    @property
    def clock_rate(self) -> float:
        """
        Return the core clock rate in Hz (CLOCK_RATE is in kHz for many devices).
        """
        return self.attrs['CLOCK_RATE'] * 1e3

    @property
    def memory_bandwidth(self) -> float:
        """
        Approx memory BW (bytes/s), from memory clock & bus width.
        """
        mem_clock_hz = self.attrs['MEMORY_CLOCK_RATE'] * 1e3
        bus_width_bits = self.attrs['GLOBAL_MEMORY_BUS_WIDTH']
        return (bus_width_bits * mem_clock_hz * 2) / 8.0


#############################################################################
# 2) PTX ANALYSIS
#############################################################################

@dataclass
class KernelAnalysis:
    """
    We specifically track:
      - coalesced loads/stores: mem_insts_coal
      - uncoalesced loads/stores: mem_insts_uncoal
      - total compute instructions (for eqn(19))
      - synch_insts for eqn(26)/(27)
    """
    mem_insts_coal: int
    mem_insts_uncoal: int
    comp_insts: int
    synch_insts: int
    shared_mem: int
    registers_per_thread: int

class PTXAnalyzer:
    """
    Simple PTX code analysis that tries to guess which memory ops might be coalesced vs. uncoalesced.
    This is an approximation: if we see big strides, we mark them as uncoalesced, else coalesced.
    """
    def __init__(self, ptx_code: str):
        self.ptx_code = ptx_code

    def analyze(self) -> KernelAnalysis:
        (coal_count, uncoal_count) = self._distinguish_coal_uncoal()
        comp_count = self._count_comp_ops()
        synch_count = self.ptx_code.count('bar.sync')
        shmem = self._analyze_shared_mem()
        rprthr = self._analyze_registers()

        return KernelAnalysis(
            mem_insts_coal=coal_count,
            mem_insts_uncoal=uncoal_count,
            comp_insts=comp_count,
            synch_insts=synch_count,
            shared_mem=shmem,
            registers_per_thread=rprthr
        )

    def _distinguish_coal_uncoal(self):
        """
        Quick heuristic: if the instruction looks like `ld.global [reg + small_offset]`, treat as coalesced,
        else uncoalesced. We combine loads & stores in one pot, for simplicity.
        """
        # find all ld.global or st.global lines
        mem_ops = re.findall(r'(ld\.global|st\.global)\s.*?\[(%\w+)\s*(\+|\-)\s*(\d+)\]', self.ptx_code)
        if not mem_ops:
            return (0,0)
        coal_count=0
        uncoal_count=0
        for (op, reg, sign, offset_str) in mem_ops:
            offval = abs(int(offset_str))
            if offval <= 128:
                coal_count +=1
            else:
                uncoal_count +=1
        return (coal_count, uncoal_count)

    def _count_comp_ops(self):
        """
        For Hong & Kim, we just need total non-memory instructions. We'll assume each .f32 .s32 .f64 is '1' comp inst.
        """
        # count lines that are likely arithmetic or logic
        # This is naive. We skip ld, st, bar.sync. We'll treat everything else as 'comp'.
        lines = self.ptx_code.split('\n')
        comp=0
        for ln in lines:
            ln_strip = ln.strip()
            if ('ld.' in ln_strip) or ('st.' in ln_strip) or ('bar.sync' in ln_strip):
                continue
            # if something has an opcode like add, mul, etc:
            if re.search(r'\b(add|mul|mad|fma|sub|div|neg|abs|and|or|not|xor)\b', ln_strip):
                comp+=1
            # also if we see .f32 or .s32 in an instruction line, call it comp
            elif re.search(r'\.(f32|s32|u32|f64|s64)', ln_strip):
                comp+=1
        return comp

    def _analyze_shared_mem(self) -> int:
        shared_vars = re.findall(
            r'\.shared\s+\.(\w+)\s+(\w+)\s*\[(\d+)\]', 
            self.ptx_code
        )
        type_sizes = {'u32':4, 's32':4, 'f32':4, 'u64':8, 'f64':8}
        total_bytes = 0
        for (dtype, varname, count) in shared_vars:
            total_bytes += int(count)* type_sizes.get(dtype.strip(),4)
        return total_bytes

    def _analyze_registers(self) -> int:
        regs = re.findall(r'\.reg\s+\.(b32|s32|u32|f32|b64|s64|u64|f64)\s+%\w+<(\d+)>', self.ptx_code)
        if not regs:
            return 32
        # pick largest
        return max(int(x[1]) for x in regs)


#############################################################################
# 3) ExecutionTimeEstimator (Hong & Kim Equations)
#############################################################################

class ExecutionTimeEstimator:
    def __init__(self,
                 analysis: KernelAnalysis,
                 grid: Tuple[int,int],
                 block: Tuple[int,int],
                 arch: GPUArchitecture):
        self.analysis = analysis
        self.grid = grid
        self.block = block
        self.arch = arch

        # We'll parse arch.default_params as well
        p = arch.default_params
        self.Mem_LD                = p['Mem_LD']                # base round-trip in sec
        self.Departure_del_coal    = p['Departure_del_coal']
        self.Departure_del_uncoal  = p['Departure_del_uncoal']
        self.issue_cycles          = p['issue_cycles']           # 4 cycles/instruction

        self.sm_count      = arch.sm_count
        self.clock_rate_hz = arch.clock_rate
        self.mem_bandwidth = arch.memory_bandwidth  # bytes / sec

    def estimate_time(self) -> float:
        """
        Return total kernel runtime in *nanoseconds*, following 
        Hong & Kim eqns (22),(23),(24).
        """

        # Step 1) Convert everything into warp-level values
        #   #coalesced_insts, #uncoalesced_insts
        n_coal  = self.analysis.mem_insts_coal
        n_uncoal= self.analysis.mem_insts_uncoal
        mem_total = (n_coal + n_uncoal)

        # Step 2) Weighted memory-latency for a single warp
        #   eqn(10),(11)
        #   Mem_L_uncoal = Mem_LD + (32-1)*Departure_del_uncoal
        #   Mem_L_coal   = Mem_LD
        Mem_L_uncoal = self.Mem_LD + (32-1)* self.Departure_del_uncoal
        Mem_L_coal   = self.Mem_LD

        if mem_total>0:
            w_uncoal = n_uncoal / float(mem_total)
            w_coal   = n_coal   / float(mem_total)
        else:
            w_uncoal = 0.0
            w_coal   = 0.0

        # eqn(12): Mem_L = w_uncoal*Mem_L_uncoal + w_coal*Mem_L_coal
        Mem_L = w_uncoal*Mem_L_uncoal + w_coal*Mem_L_coal

        # Also eqn(15): departure_delay = ...
        # For uncoalesced we do (32 * departure_del_uncoal), for coalesced we do (1 * departure_del_coal).
        # eqn(15) => ( (#Uncoal_per_mw)* departure_del_uncoal ) * weight_uncoal + departure_del_coal * weight_coal
        dd_uncoal = 32*self.Departure_del_uncoal
        dd_coal   = self.Departure_del_coal
        departure_delay = dd_uncoal*w_uncoal + dd_coal*w_coal

        # Step 3) MWP_Without_BW_full = Mem_L / departure_delay
        #   clamp to #Active_warps_per_SM as eqn(16),(17)
        active_warps_per_sm, rep = self._compute_active_warps_and_reps()

        if departure_delay>1e-14:  # avoid zero-div
            MWP_woBW_full = Mem_L / departure_delay
        else:
            MWP_woBW_full = active_warps_per_sm

        MWP_woBW = min(MWP_woBW_full, active_warps_per_sm)

        # Step 4) MWP_peak_BW eqn(6)
        #   BW_per_warp = freq * load_bytes_per_warp / Mem_L. 
        #   Let's assume 128 bytes per warp access for float4 or so
        load_bytes_per_warp = 128.0
        # freq is clock_rate_hz. But we need consistent units. The paper used cycles for freq.
        # We'll do everything in "sec" here, so let's carefully do:
        #   BW_per_warp = (clock_rate_hz * load_bytes_per_warp) / (1 / Mem_L)
        # Actually, eqn(31) => BW_per_warp = freq * load_bytes_per_warp / Mem_L 
        # So let's do that in "bytes/sec".
        # freq is cycles/sec, multiply by load_bytes => bytes/sec, then / Mem_L => 
        # careful about unit of Mem_L => seconds. So that yields bytes/sec if Mem_L is in sec? 
        # Actually simpler to do exactly eqn(31):
        if Mem_L>1e-14:
            BW_per_warp = (self.clock_rate_hz * load_bytes_per_warp) / (1.0/self.Mem_LD)
            # but the paper's eqn(7) is a bit abstract. We'll do a simpler approach:
            # If we want to replicate eqn(7) strictly, we do: freq*(bytes)/Mem_L in cycles?
            # We'll approximate:
            BW_per_warp = (self.clock_rate_hz * load_bytes_per_warp)/(1.0/ Mem_L)
        else:
            BW_per_warp = 1e9

        # eqn(6): MWP_peak_BW = Mem_Bandwidth / (BW_per_warp * #ActiveSM)
        # Mem_Bandwidth is bytes/sec. 
        MWP_peak_BW = self.mem_bandwidth / (BW_per_warp * self.sm_count)

        # Step 5) final MWP
        MWP = min(MWP_woBW, MWP_peak_BW, active_warps_per_sm)

        # Step 6) compute Mem_cycles, Comp_cycles for eqn(18),(19)
        # eqn(19): comp_cycles = #Issue_cycles*(#total_insts).
        #   #total_insts => comp_insts + synch_insts + mem_insts_coal + mem_insts_uncoal
        total_insts = (self.analysis.comp_insts 
                       + self.analysis.synch_insts 
                       + mem_total)
        Comp_cycles = total_insts * self.issue_cycles

        # eqn(18):
        #   Mem_cycles = Mem_L_uncoal * #Uncoal_Mem_insts + Mem_L_coal * #Coal_Mem_insts
        #   in *cycles*, so we multiply seconds by clock_rate_hz
        #   => (Mem_L_uncoal in s) * clock_rate_hz => cycles for uncoalesced 
        #   times #Uncoal_Mem_insts
        mem_uncoal_cycles = (Mem_L_uncoal*self.clock_rate_hz)* float(n_uncoal)
        mem_coal_cycles   = (Mem_L_coal  *self.clock_rate_hz)* float(n_coal)
        Mem_cycles = mem_uncoal_cycles + mem_coal_cycles

        # Step 7) compute CWP eqn(8),(9)
        # eqn(8): CWP_full = (Mem_cycles + Comp_cycles)/Comp_cycles
        if Comp_cycles>1:
            CWP_full = (Mem_cycles + Comp_cycles)/ Comp_cycles
        else:
            CWP_full = active_warps_per_sm
        CWP = min(CWP_full, active_warps_per_sm)

        # Step 8) Decide which eqn(22),(23),(24) to use
        N = active_warps_per_sm
        # #Rep is rep
        # eqn(22): if (MWP == N) and (CWP == N)
        # eqn(23): if (CWP >= MWP) or (Comp_cycles > Mem_cycles)
        # eqn(24): else (MWP > CWP)
        if (abs(MWP - N)<0.001) and (abs(CWP -N)<0.001):
            # eqn(22)
            # (Mem_cycles + Comp_cycles + (Comp_cycles/#Mem_insts)*(MWP-1)) * rep
            # #Mem_insts = mem_total
            if mem_total>0:
                comp_p = Comp_cycles/ float(mem_total)
            else:
                comp_p = 0.0
            time_per_rep_cycles = (Mem_cycles 
                                   + Comp_cycles 
                                   + comp_p*(MWP-1))
        elif (CWP >= MWP) or (Comp_cycles> Mem_cycles):
            # eqn(23)
            # (Mem_cycles*(N/MWP) + (Comp_cycles/#Mem_insts)*(MWP-1)) * rep
            if mem_total>0:
                comp_p = Comp_cycles/ float(mem_total)
            else:
                comp_p = 0.0
            time_per_rep_cycles = (Mem_cycles*(N/ MWP)
                                   + comp_p*(MWP-1))
        else:
            # eqn(24)
            # Mem_L + (Comp_cycles*N)
            # in cycles: we should do Mem_L in cycles => Mem_L*s * clock_rate_hz
            Mem_L_cycles = Mem_L* self.clock_rate_hz
            time_per_rep_cycles = Mem_L_cycles + (Comp_cycles*N)

        # Step 9) synchronization overhead eqn(26),(27):
        # Synch_cost = departure_delay*(MWP-1)* #synch_insts* #Active_blocks_per_SM * rep
        # departure_delay is in sec, multiply by clock_rate => cycles
        # #Active_blocks_per_SM is from occupancy
        (active_blocks_per_sm, _) = self._compute_active_warps_and_reps()
        synch_cost_cycles = 0.0
        if self.analysis.synch_insts>0:
            dep_delay_cycles = departure_delay*self.clock_rate_hz
            synch_cost_cycles = dep_delay_cycles*(MWP-1)* self.analysis.synch_insts* active_blocks_per_sm* rep

        total_cycles = (time_per_rep_cycles* rep) + synch_cost_cycles

        # convert cycles => ns
        total_ns = total_cycles / self.clock_rate_hz * 1e9
        return total_ns

    def _compute_active_warps_and_reps(self) -> Tuple[float,float]:
        """
        Returns (active_warps_per_sm, #Rep).
        #Rep = how many times the SM must repeat blocks.
        """
        block_threads = self.block[0]* self.block[1]
        # naive occupancy-based:
        max_threads_per_sm = self.arch.attrs['MAX_THREADS_PER_MULTIPROCESSOR']
        max_blocks = max_threads_per_sm // block_threads
        if max_blocks<1:
            max_blocks=1
        # also check shared mem usage
        if self.analysis.shared_mem>0:
            blocks_by_shmem = self.arch.attrs['MAX_SHARED_MEMORY_PER_MULTIPROCESSOR']// self.analysis.shared_mem
            if blocks_by_shmem<1:
                blocks_by_shmem=1
            max_blocks = min(max_blocks, blocks_by_shmem)

        # also check register usage if you want
        # ...
        # We'll skip for brevity

        # active warps per SM
        warps_per_block = math.ceil(block_threads/32.0)
        active_warps_per_sm = max_blocks* warps_per_block

        # total blocks
        total_blocks = self.grid[0]* self.grid[1]
        # #Rep = total_blocks / (self.sm_count* max_blocks)
        # but we must handle partial leftover => use math.ceil
        # We'll do an integer or float version:
        rep = math.ceil(total_blocks/(self.sm_count* max_blocks))

        return (float(active_warps_per_sm), float(rep))


#############################################################################
# 4) UTILITY: Compile & Benchmark
#############################################################################

def compile_kernel(path:str, arch_options:str):
    with open(path,'r') as f:
        source = f.read()
    ptx = compile(source, target="ptx", no_extern_c=True, options=arch_options).decode()
    match = re.search(r'\.entry\s+(\w+)', ptx)
    if not match:
        raise RuntimeError("Could not find .entry kernel in PTX!")
    kernel_name = match.group(1)
    mod = SourceModule(source, no_extern_c=True, options=arch_options)
    with open(path+".ptx",'w') as ff:
        ff.write(ptx)
    return mod, ptx, kernel_name

def benchmark_kernel(kernel_func, args, grid:Tuple[int,int], block:Tuple[int,int], runs:int=50)->float:
    start= cuda.Event(); end= cuda.Event()
    block_3d= (block[0], block[1],1)
    grid_3d= (grid[0], grid[1],1)
    # warmup
    for _ in range(10):
        kernel_func(*args, block=block_3d, grid=grid_3d)
    times=[]
    for _ in range(runs):
        start.record()
        kernel_func(*args, block=block_3d, grid=grid_3d)
        end.record()
        end.synchronize()
        times.append(start.time_till(end)*1e6)  # microseconds
    return float(np.median(times))

#############################################################################
# 5) MAIN
#############################################################################

if __name__=="__main__":
    if len(sys.argv)!=6:
        print("Usage: python time_model.py <kernel.cu> <grid_x> <block_x> <data_size> <runs>")
        sys.exit(1)

    kernel_path= sys.argv[1]
    grid_size= int(sys.argv[2])
    block_size= int(sys.argv[3])
    data_size= int(sys.argv[4])
    runs= int(sys.argv[5])

    # 1) Create the arch
    arch= GPUArchitecture()
    (maj, minr) = arch.compute_capability
    arch_options = [f"-arch=sm_{maj}{minr}", "--ptxas-options=-v"]

    # 2) Compile
    mod, ptx, kernel_name = compile_kernel(kernel_path, arch_options)
    kernel_func= mod.get_function(kernel_name)

    # 3) Prepare data
    a_np= np.random.randn(data_size).astype(np.float32)
    b_np= np.random.randn(data_size).astype(np.float32)
    d_a= cuda.mem_alloc(a_np.nbytes)
    d_b= cuda.mem_alloc(b_np.nbytes)
    d_c= cuda.mem_alloc(a_np.nbytes)
    cuda.memcpy_htod(d_a,a_np)
    cuda.memcpy_htod(d_b,b_np)
    args= [d_a,d_b,d_c, np.int32(data_size)]

    # 4) PTX analysis
    analysis_obj= PTXAnalyzer(ptx)
    analysis= analysis_obj.analyze()

    # 5) Construct the estimator & estimate
    estimator= ExecutionTimeEstimator(
        analysis=analysis,
        grid=(grid_size,1),
        block=(block_size,1),
        arch=arch
    )
    estimated_ns= estimator.estimate_time()

    # 6) Actual benchmark
    actual_ns= benchmark_kernel(kernel_func, args, (grid_size,1), (block_size,1), runs)
    # our benchmark_kernel returns microseconds => multiply by 1e3 => ns
    actual_ns*=1e3

    diff= abs(estimated_ns- actual_ns)/ max(actual_ns,1e-9)*100.0
    print(f"Kernel: {kernel_name}")
    print(f"EstimatedTime(ns): {estimated_ns:.2f}")
    print(f"ActualTime(ns):    {actual_ns:.2f}")
    print(f"Diff(%%):          {diff:.2f}")
