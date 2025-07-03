#!/usr/bin/env python3
import re
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

class PTXAnalyzer:
    """
    Enhanced PTX analyzer that:
      1. Accurately separates the header and body of the .entry function.
      2. Counts global memory instructions ("ld.global" and "st.global") from the entire function
         body as a fallback if basic block splitting misses them.
      3. Improves coalescing estimation by analyzing memory access patterns alongside block_x.
      4. Builds a robust CFG to detect loops and scale instruction counts by estimated iterations.
      5. Enhances instruction categorization with precise regular expressions.

    For small block_x values, a global scan ensures non-zero memory operation counts.
    """

    def __init__(self, ptx_code: str, ptxas_log: str, arch, block_x: int, block_y: int, config: dict):
        self.ptx_code = ptx_code
        self.ptxas_log = ptxas_log
        self.arch = arch
        self.block_x = block_x
        self.block_y = block_y
        self.config = config or {}

        # Defaults
        self.regs_per_thread = 32
        self.shared_mem_used = 0

        # Calibration data
        self.calib = arch.calibration_data if arch.calibration_data else {}

        # Data structures for analysis
        self.labels: Dict[str, int] = {}
        self.cfg: Dict[int, List[int]] = defaultdict(list)
        self.basic_blocks: List[List[str]] = []
        self.line_to_block: Dict[int, int] = {}

        self.occupancy_factor = 1.0
        self.shared_bank_conflict_factor = 1.0

    def analyze(self):
        self._parse_ptxas_info()
        lines = self.ptx_code.split('\n')
        self._collect_labels(lines)
        self._build_basic_blocks(lines)
        self._build_cfg()
        loops = self._detect_loops()
        loop_iterations = self._estimate_loop_iterations(loops)

        self.occupancy_factor = self._estimate_occupancy_factor()
        self.shared_bank_conflict_factor = self._estimate_shared_bank_conflicts()

        # Global scan for memory ops as a fallback
        global_ld, global_st = self._count_global_ops_global()

        # Count instructions in basic blocks
        (blk_ld, blk_st, loc_cnt, shr_cnt, sync_cnt,
         fp_cnt, int_cnt, sfu_cnt, alu_cnt) = self._accumulate_block_insts(loop_iterations)

        # Fallback to global counts if basic blocks miss global memory ops
        if (blk_ld + blk_st) == 0:
            blk_ld, blk_st = global_ld, global_st

        base_global = blk_ld + blk_st

        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        effective_block_x = self.block_x
        mem_coal, mem_un, mem_part = self._coalescing_breakdown(effective_block_x, base_global)

        total_compute = fp_cnt + int_cnt + sfu_cnt + alu_cnt
        total_insts = (mem_coal + mem_un + mem_part +
                       loc_cnt + shr_cnt + sync_cnt + total_compute)

        from cuda_kernel_energy_empirical.experiments.rq2.gpu_common import KernelAnalysis
        return KernelAnalysis(
            mem_coal=int(mem_coal),
            mem_uncoal=int(mem_un),
            mem_partial=int(mem_part),
            local_insts=int(loc_cnt),
            shared_insts=int(shr_cnt),
            synch_insts=int(sync_cnt),
            fp_insts=int(fp_cnt),
            int_insts=int(int_cnt),
            sfu_insts=int(sfu_cnt),
            alu_insts=int(alu_cnt),
            total_insts=int(total_insts),
            registers_per_thread=self.regs_per_thread,
            shared_mem_bytes=self.shared_mem_used,
            block_x=self.block_x,
            block_y=self.block_y
        )

    ### Parsing PTXAS Log Info
    def _parse_ptxas_info(self):
        reg_pat = re.compile(r"Used\s+(\d+)\s+registers")
        smem_pat = re.compile(r"Used\s+(\d+)\s+bytes\s+shared", re.IGNORECASE)
        for ln in self.ptxas_log.split('\n'):
            if m := reg_pat.search(ln):
                self.regs_per_thread = int(m.group(1))
            if m := smem_pat.search(ln):
                self.shared_mem_used = int(m.group(1))

    ### Basic Block Splitting and CFG Construction
    def _collect_labels(self, lines: List[str]):
        label_pat = re.compile(r"^\s*([a-zA-Z_$.][\w$.]*):")
        for i, ln in enumerate(lines):
            if m := label_pat.match(ln.strip()):
                self.labels[m.group(1)] = i

    def _build_basic_blocks(self, lines: List[str]):
        current_block = []
        block_id = 0
        control_flow_pat = re.compile(r"\b(?:bra(?:\.uni)?|ret|exit)\b", re.IGNORECASE)

        for i, ln in enumerate(lines):
            ln = ln.strip()
            if not ln:
                continue
            if self.labels.get(ln[:-1]) is not None:  # Label starts a new block
                if current_block:
                    self.basic_blocks.append(current_block)
                    block_id += 1
                current_block = [ln]
                self.line_to_block[i] = block_id
            else:
                current_block.append(ln)
                self.line_to_block[i] = block_id
                if control_flow_pat.search(ln):  # End block on control flow
                    self.basic_blocks.append(current_block)
                    current_block = []
                    block_id += 1
        if current_block:
            self.basic_blocks.append(current_block)

    def _build_cfg(self):
        for bid, block in enumerate(self.basic_blocks):
            last_line = block[-1].strip()
            if m := re.search(r"\bbra(?:\.uni)?\s+([a-zA-Z_$.][\w$.]*)", last_line):
                target = m.group(1)
                if target in self.labels:
                    self.cfg[bid].append(self.line_to_block[self.labels[target]])
            elif m := re.search(r"@(!?%[a-zA-Z_$.][\w$.]*)\s+bra\s+([a-zA-Z_$.][\w$.]*)", last_line):
                target = m.group(2)
                if target in self.labels:
                    self.cfg[bid].append(self.line_to_block[self.labels[target]])
                if bid + 1 < len(self.basic_blocks):
                    self.cfg[bid].append(bid + 1)  # Fall-through
            elif not re.search(r"\b(?:ret|exit)\b", last_line) and bid + 1 < len(self.basic_blocks):
                self.cfg[bid].append(bid + 1)  # Fall-through

    def _detect_loops(self) -> List[Tuple[int, int]]:
        visited = set()
        stack = []
        loops = []

        def dfs(u):
            visited.add(u)
            stack.append(u)
            for v in self.cfg[u]:
                if v not in visited:
                    dfs(v)
                elif v in stack:
                    loops.append((v, u))
            stack.pop()

        if self.basic_blocks:
            dfs(0)
        return loops

    def _estimate_loop_iterations(self, loops: List[Tuple[int, int]]) -> Dict[int, int]:
        loop_iters = {}
        user_loops = self.config.get("loop_iterations", {})
        for head, tail in loops:
            b_line = self._block_start_line(head)
            labelname = self._line_if_label(b_line)
            if labelname and labelname in user_loops:
                loop_iters[head] = user_loops[labelname]
            else:
                # Heuristic: use block_x as default iteration count
                loop_iters[head] = self.block_x
        return loop_iters

    def _block_start_line(self, b_idx: int) -> int:
        return next((i for i, blk in self.line_to_block.items() if blk == b_idx), -1)

    def _line_if_label(self, line_idx: int) -> str:
        return next((lbl for lbl, li in self.labels.items() if li == line_idx), None)

    ### Occupancy and Shared Memory Conflict Estimation
    def _estimate_occupancy_factor(self) -> float:
        shape_factor = float(self.calib.get("shape_occupancy_factor", 0.2))
        da = self.arch.attrs
        max_thr_sm = da.get('MAX_THREADS_PER_MULTIPROCESSOR', 1536)
        thr_block = self.block_x * self.block_y
        if thr_block == 0:
            return 1.0
        tlim = max_thr_sm // thr_block
        warps_per_block = math.ceil(thr_block / da.get('WARP_SIZE', 32))
        total_warps_psm = tlim * warps_per_block
        ideal_warps = 32.0
        ratio = min(max(total_warps_psm / ideal_warps, 0.25), 1.5)
        aspect_ratio = self.block_x / self.block_y if self.block_y > 0 else 1.0
        final_val = ratio - shape_factor * math.log(max(aspect_ratio, 1e-6))
        return max(0.1, min(final_val, 2.0))

    def _estimate_shared_bank_conflicts(self) -> float:
        base_conflict = float(self.calib.get("base_bank_conflict", 1.0))
        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        return base_conflict * 1.2 if self.block_x > warp_size else base_conflict

    ### Global Memory Operation Counting
    def _count_global_ops_global(self) -> Tuple[int, int]:
        entry_match = re.search(r"\.entry\s+\w+\s*\((.*?)\)\s*{(.*)}", self.ptx_code, re.DOTALL)
        body = entry_match.group(2) if entry_match else self.ptx_code
        ld_ops = len(re.findall(r"\bld\.global\b", body, re.IGNORECASE))
        st_ops = len(re.findall(r"\bst\.global\b", body, re.IGNORECASE))
        return ld_ops, st_ops

    ### Instruction Counting in Basic Blocks
    def _accumulate_block_insts(self, loop_iters: Dict[int, int]) -> Tuple[int, ...]:
        num_blocks = len(self.basic_blocks)
        counts = {k: [0] * num_blocks for k in 
                  ['ld', 'st', 'loc', 'shr', 'sync', 'fp', 'int', 'sfu', 'alu']}

        for b_idx in range(num_blocks):
            counts_tuple = self._count_block_insts(b_idx)
            for key, val in zip(counts.keys(), counts_tuple):
                counts[key][b_idx] = val

        visited = set()
        queue = deque([0])
        block_loop_map = self._find_block_loop_membership(loop_iters)
        totals = {k: 0 for k in counts}

        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            iteration_factor = 1
            for lh in block_loop_map[cur]:
                iteration_factor *= loop_iters.get(lh, 1)
            for k in totals:
                totals[k] += counts[k][cur] * iteration_factor
            for nxt in self.cfg[cur]:
                if nxt not in visited:
                    queue.append(nxt)

        return tuple(totals[k] for k in ['ld', 'st', 'loc', 'shr', 'sync', 'fp', 'int', 'sfu', 'alu'])

    def _count_block_insts(self, b_idx: int) -> Tuple[int, ...]:
        block_lines = self.basic_blocks[b_idx]
        ldg = stg = loc = shr = sy = fpc = inc = sfc = alc = 0

        patterns = {
            'fp': re.compile(r"\b(fadd|fsub|fmul|fma|fdiv|frcp|fsqrt|frsqrt|fsin|fcos)\b", re.IGNORECASE),
            'int': re.compile(r"\b(add|sub|mul|div|rem|min|max|popc|clz)\b", re.IGNORECASE),
            'sfu': re.compile(r"\b(sin|cos|ex2|lg2|rcp|rsqrt|sqrt)\b", re.IGNORECASE),
            'mem': re.compile(r"\b(ld|st)\.(global|local|shared)\b", re.IGNORECASE),
            'sync': re.compile(r"\bbar\.sync\b", re.IGNORECASE)
        }

        for ln in block_lines:
            ln_lower = ln.lower()
            if "ld.global" in ln_lower:
                ldg += 1
            elif "st.global" in ln_lower:
                stg += 1
            elif patterns['mem'].search(ln) and "local" in ln_lower:
                loc += 1
            elif patterns['mem'].search(ln) and "shared" in ln_lower:
                shr += 1
            elif patterns['sync'].search(ln):
                sy += 1
            elif patterns['fp'].search(ln):
                fpc += 1
            elif patterns['int'].search(ln):
                inc += 1
            elif patterns['sfu'].search(ln):
                sfc += 1
            else:
                alc += 1  # Default to ALU for unclassified instructions

        return (ldg, stg, loc, shr, sy, fpc, inc, sfc, alc)

    def _find_block_loop_membership(self, loop_iters: Dict[int, int]) -> Dict[int, Set[int]]:
        block_map = defaultdict(set)
        for head in loop_iters:
            visited = set()
            queue = deque([head])
            while queue:
                c = queue.popleft()
                if c in visited:
                    continue
                visited.add(c)
                block_map[c].add(head)
                for nxt in self.cfg[c]:
                    if nxt != head:
                        queue.append(nxt)
        return block_map

    ### Coalescing Estimation
    def _coalescing_breakdown(self, effective_block_x: int, global_ops: float) -> Tuple[float, float, float]:
        """
        Now we model partial coalescing. 
        We'll parse the 'partial_coalesce_slope' from self.calib, 
        and let some fraction of global_ops become 'mem_partial'.
        The remainder splits into mem_coal vs. mem_uncoal. 
        """
        if global_ops <= 0:
            return (0.0, 0.0, 0.0)

        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        stride_factor = self._analyze_memory_strides()  # existing function
        max_coalesced = min(effective_block_x / float(warp_size), 1.0)
        # This is the fraction that truly gets coalesced:
        actual_coalesced = max_coalesced * stride_factor

        # We'll treat partial_coalescing as the fraction in-between
        # 'actual_coalesced' and a slope-based guess from calibration
        partial_slope = float(self.calib.get("partial_coalesce_slope", 0.0))
        partial_intcp = float(self.calib.get("partial_coalesce_intercept", 0.0))

        # Estimate partial fraction using a linear function of block_x offset
        # for demonstration, say partial_fraction ~ slope*(effective_block_x) + intercept
        guess_partial = partial_slope * effective_block_x + partial_intcp
        guess_partial = max(0.0, min(guess_partial, 1.0))  # clamp 0..1

        # The fraction that is definitely 'fully coalesced'
        mem_coal_fraction = actual_coalesced * (1.0 - guess_partial)
        # The fraction that is partial
        mem_part_fraction = guess_partial * (1.0 - actual_coalesced)
        # The fraction that is uncoalesced
        mem_un_fraction   = 1.0 - (mem_coal_fraction + mem_part_fraction)

        mem_coal = global_ops * mem_coal_fraction
        mem_part = global_ops * mem_part_fraction
        mem_un   = global_ops * mem_un_fraction

        return (mem_coal, mem_un, mem_part)


    def _analyze_memory_strides(self) -> float:
        stride1_count = 0
        total_mem_ops = 0
        stride_pat = re.compile(r"\[\s*%r\d+\s*\+\s*(\d+)\s*\]")

        for block in self.basic_blocks:
            for ln in block:
                if 'ld.global' in ln or 'st.global' in ln:
                    total_mem_ops += 1
                    if m := stride_pat.search(ln):
                        offset = int(m.group(1))
                        if offset <= 4:  # Assume stride-1 for small offsets
                            stride1_count += 1

        return stride1_count / total_mem_ops if total_mem_ops > 0 else 1.0

# Example usage
# ptx_analyzer = PTXAnalyzer(ptx_code, ptxas_log, arch, block_x, block_y, config)
# analysis = ptx_analyzer.analyze()