#!/usr/bin/env python3
import re
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set

class PTXAnalyzer:
    """
    Improved PTX analyzer that:
      1. Separates the header and the body of the .entry function.
      2. Counts global memory instructions (“ld.global” and “st.global”) 
         from the entire function body if basic block splitting does not capture them.
      3. Uses a more robust method for coalescing estimation based on the actual
         effective “X-dimension” of the thread block.
      4. (Optionally) Builds a simple CFG to detect loops and scale counts by iteration.
    
    For kernels that use a very small block_x, we now fall back to a “global scan”
    so that we don’t get 0 counts.
    """

    def __init__(self, 
                 ptx_code: str, 
                 ptxas_log: str, 
                 arch, 
                 block_x: int, 
                 block_y: int, 
                 config: dict):
        self.ptx_code = ptx_code
        self.ptxas_log = ptxas_log
        self.arch = arch
        self.block_x = block_x
        self.block_y = block_y
        self.config = config or {}

        # Defaults
        self.regs_per_thread = 32
        self.shared_mem_used = 0

        # For convenience, store calibration data in a local var:
        self.calib = arch.calibration_data if arch.calibration_data else {}

        self.labels: Dict[str, int] = {}
        self.cfg: Dict[int, List[int]] = {}
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

        # Scan code for global ops
        global_ld, global_st = self._count_global_ops_global()

        # Basic block instruction counts
        (blk_ld, blk_st, loc_cnt, shr_cnt, sync_cnt,
         fp_cnt, int_cnt, sfu_cnt, alu_cnt) = self._accumulate_block_insts(loop_iterations)

        # Fallback if no blocks see global mem
        if (blk_ld + blk_st) == 0:
            blk_ld, blk_st = global_ld, global_st

        base_global = (blk_ld + blk_st)

        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        effective_block_x = max(self.block_x, warp_size)
        (mem_coal, mem_un, mem_part) = self._coalescing_breakdown(effective_block_x, base_global)

        total_compute = fp_cnt + int_cnt + sfu_cnt + alu_cnt
        total_insts = (mem_coal + mem_un + mem_part +
                       loc_cnt + shr_cnt + sync_cnt + total_compute)

        from gpu_common import KernelAnalysis
        return KernelAnalysis(
            mem_coal = int(mem_coal),
            mem_uncoal = int(mem_un),
            mem_partial = int(mem_part),
            local_insts = int(loc_cnt),
            shared_insts = int(shr_cnt),
            synch_insts = int(sync_cnt),
            fp_insts = int(fp_cnt),
            int_insts = int(int_cnt),
            sfu_insts = int(sfu_cnt),
            alu_insts = int(alu_cnt),
            total_insts = int(total_insts),
            registers_per_thread = self.regs_per_thread,
            shared_mem_bytes = self.shared_mem_used,
            block_x = self.block_x,
            block_y = self.block_y
        )

    # ---------------------------
    # Parsing ptxas log info
    # ---------------------------
    def _parse_ptxas_info(self):
        reg_pat = re.compile(r"Used\s+(\d+)\s+registers")
        smem_pat = re.compile(r"Used\s+(\d+)\s+bytes\s+shared", re.IGNORECASE)
        for ln in self.ptxas_log.split('\n'):
            m1 = reg_pat.search(ln)
            if m1:
                self.regs_per_thread = int(m1.group(1))
            m2 = smem_pat.search(ln)
            if m2:
                self.shared_mem_used = int(m2.group(1))

    # ---------------------------
    # Basic block splitting (as before)
    # ---------------------------
    def _collect_labels(self, lines: List[str]):
        label_pat = re.compile(r"^([.\w\$L]+[0-9]*):")
        for i, ln in enumerate(lines):
            ln_stripped = ln.strip()
            m = label_pat.match(ln_stripped)
            if m:
                label_name = m.group(1)
                self.labels[label_name] = i

    def _build_basic_blocks(self, lines: List[str]):
        current_block = []
        block_id = 0
        i = 0
        n_lines = len(lines)
        while i < n_lines:
            ln = lines[i].rstrip()
            if ln.strip() == "":
                i += 1
                continue
            if re.match(r"^([.\w\$L]+[0-9]*):", ln):
                if current_block:
                    self.basic_blocks.append(current_block)
                    current_block = []
                    block_id += 1
                current_block.append(ln)
                self.line_to_block[i] = block_id
            else:
                current_block.append(ln)
                self.line_to_block[i] = block_id
                if re.search(r"\bbra\b", ln) and not re.search(r"@!", ln):
                    self.basic_blocks.append(current_block)
                    current_block = []
                    block_id += 1
            i += 1
        if current_block:
            self.basic_blocks.append(current_block)

    def _build_cfg(self):
        num_blocks = len(self.basic_blocks)
        self.cfg = {i: [] for i in range(num_blocks)}
        for b_idx in range(num_blocks):
            if not self.basic_blocks[b_idx]:
                continue
            last_line = self.basic_blocks[b_idx][-1].strip()
            m = re.search(r"\bbra\s+(\S+)", last_line)
            if m:
                lbl = m.group(1)
                if lbl in self.labels:
                    tgt_line = self.labels[lbl]
                    tgt_block = self.line_to_block[tgt_line]
                    self.cfg[b_idx].append(tgt_block)
                continue
            m2 = re.search(r"\b@!?p.*\bbra\s+(\S+)", last_line)
            if m2:
                lbl = m2.group(1)
                if lbl in self.labels:
                    tline = self.labels[lbl]
                    tblock = self.line_to_block[tline]
                    self.cfg[b_idx].append(tblock)
                if b_idx < num_blocks - 1:
                    self.cfg[b_idx].append(b_idx+1)
                continue
            if not re.search(r"\bret\b", last_line):
                if b_idx < num_blocks - 1:
                    self.cfg[b_idx].append(b_idx+1)

    def _detect_loops(self) -> List[Tuple[int,int]]:
        visited = set()
        stack = []
        loops = []
        def dfs(u):
            visited.add(u)
            stack.append(u)
            for v in self.cfg[u]:
                if v not in visited:
                    dfs(v)
                else:
                    if v in stack:
                        loops.append((v, u))
            stack.pop()
        if 0 in self.cfg:
            dfs(0)
        return loops

    def _estimate_loop_iterations(self, loops: List[Tuple[int,int]]) -> Dict[int,int]:
        loop_iters = {}
        user_loops = self.config.get("loop_iterations", {})
        for (head, tail) in loops:
            iteration_count = 1
            b_line = self._block_start_line(head)
            labelname = self._line_if_label(b_line)
            if labelname and (labelname in user_loops):
                iteration_count = user_loops[labelname]
            else:
                dims = self.config.get("dimensions", {})
                if 'n' in dims:
                    iteration_count = dims['n']
            loop_iters[head] = iteration_count
        return loop_iters

    def _block_start_line(self, b_idx: int) -> int:
        for line_i, blk in self.line_to_block.items():
            if blk == b_idx:
                return line_i
        return -1

    def _line_if_label(self, line_idx: int) -> str:
        if line_idx < 0:
            return None
        for lbl, li in self.labels.items():
            if li == line_idx:
                return lbl
        return None

    # ---------------------------
    # Occupancy & Shared Memory Conflict Heuristics
    # ---------------------------
    def _estimate_occupancy_factor(self) -> float:
        # Instead of the old fixed ratio, read "shape_occupancy_factor" from calibration if present
        shape_factor = float(self.calib.get("shape_occupancy_factor", 0.2))
        # Possibly do a small function of block_x, block_y, shape_factor
        # We'll do the same code but multiply by shape_factor
        da = self.arch.attrs
        max_thr_sm = da['MAX_THREADS_PER_MULTIPROCESSOR']
        thr_block = self.block_x * self.block_y
        tlim = max_thr_sm // thr_block if thr_block > 0 else 1
        warps_per_block = math.ceil(thr_block / da['WARP_SIZE'])
        ideal_warps = 32.0
        blocks_per_sm = tlim
        total_warps_psm = blocks_per_sm * warps_per_block
        ratio = total_warps_psm / ideal_warps
        ratio = max(0.25, min(1.5, ratio))
        # combine with shape_factor somehow (example):
        final_val = ratio - shape_factor * math.log( max(self.block_x/self.block_y, 1e-6) )
        # clamp
        final_val = max(0.1, min(final_val, 2.0))
        return final_val

    def _estimate_shared_bank_conflicts(self) -> float:
        # We might store a "base_bank_conflict" in calibration. Otherwise fallback:
        base_conflict = float(self.calib.get("base_bank_conflict", 1.0))
        warp_size = self.arch.attrs.get("WARP_SIZE", 32)
        # example logic
        if self.block_x > warp_size:
            return base_conflict * 1.2
        else:
            return base_conflict

    # ---------------------------
    # Global scan for memory ops
    # ---------------------------
    def _count_global_ops_global(self) -> Tuple[int, int]:
        """
        Instead of relying solely on basic block splitting,
        we scan the entire PTX function body (after the .entry directive)
        for ld.global and st.global instructions.
        """
        # Extract the function body starting at the .entry directive.
        entry_match = re.search(r"\.entry\s+\w+\s*\((.*?)\)\s*{(.*)}", self.ptx_code, re.DOTALL)
        if entry_match:
            body = entry_match.group(2)
        else:
            body = self.ptx_code
        body_lower = body.lower()
        ld_ops = len(re.findall(r"ld\.global", body_lower))
        st_ops = len(re.findall(r"st\.global", body_lower))
        return ld_ops, st_ops

    # ---------------------------
    # Count instructions per basic block (using basic block splitting)
    # ---------------------------
    def _accumulate_block_insts(self, loop_iters: Dict[int,int]) -> Tuple[int,int,int,int,int,int,int,int,int]:
        num_blocks = len(self.basic_blocks)
        block_ld = [0]*num_blocks
        block_st = [0]*num_blocks
        block_local = [0]*num_blocks
        block_shared = [0]*num_blocks
        block_sync = [0]*num_blocks
        block_fp = [0]*num_blocks
        block_int = [0]*num_blocks
        block_sfu = [0]*num_blocks
        block_alu = [0]*num_blocks

        for b_idx in range(num_blocks):
            (ldg, stg, loc, shr, sy, fpc, inc, sfc, alc) = self._count_block_insts(b_idx)
            block_ld[b_idx] = ldg
            block_st[b_idx] = stg
            block_local[b_idx] = loc
            block_shared[b_idx] = shr
            block_sync[b_idx] = sy
            block_fp[b_idx] = fpc
            block_int[b_idx] = inc
            block_sfu[b_idx] = sfc
            block_alu[b_idx] = alc

        visited = set()
        queue = deque([0])
        block_loop_map = self._find_block_loop_membership(loop_iters)

        total_ld = total_st = 0
        total_local = total_shared = total_sync = 0
        total_fp = total_int = total_sfu = total_alu = 0

        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            iteration_factor = 1
            for lh in block_loop_map[cur]:
                iteration_factor *= loop_iters.get(lh, 1)
            total_ld += block_ld[cur] * iteration_factor
            total_st += block_st[cur] * iteration_factor
            total_local += block_local[cur] * iteration_factor
            total_shared += block_shared[cur] * iteration_factor
            total_sync += block_sync[cur] * iteration_factor
            total_fp += block_fp[cur] * iteration_factor
            total_int += block_int[cur] * iteration_factor
            total_sfu += block_sfu[cur] * iteration_factor
            total_alu += block_alu[cur] * iteration_factor
            for nxt in self.cfg[cur]:
                if nxt not in visited:
                    queue.append(nxt)
        return (total_ld, total_st, total_local, total_shared, total_sync,
                total_fp, total_int, total_sfu, total_alu)

    def _count_block_insts(self, b_idx: int) -> Tuple[int,int,int,int,int,int,int,int,int]:
        block_lines = self.basic_blocks[b_idx]
        ldg = stg = loc = shr = sy = fpc = inc = sfc = alc = 0
        float_pat = re.compile(r"\b(add\.f|sub\.f|mul\.f|mad\.f|fma\.f)\b", re.IGNORECASE)
        int_pat   = re.compile(r"\b(add\.s|sub\.s|mul\.s)\b", re.IGNORECASE)
        sfu_pat   = re.compile(r"\b(sin\.f|cos\.f|sqrt\.f|rsqrt\.f|rcp\.f)\b", re.IGNORECASE)
        alu_pat   = re.compile(r"\b(and\.|or\.|xor\.|shl\.|shr\.)", re.IGNORECASE)

        for ln in block_lines:
            lnl = ln.lower()
            if "ld.global" in lnl:
                ldg += 1
            if "st.global" in lnl:
                stg += 1
            if "ld.local" in lnl or "st.local" in lnl:
                loc += 1
            if "ld.shared" in lnl or "st.shared" in lnl:
                shr += 1
            if "bar.sync" in lnl:
                sy += 1
            if float_pat.search(ln):
                fpc += 1
            elif int_pat.search(ln):
                inc += 1
            elif sfu_pat.search(ln):
                sfc += 1
            elif alu_pat.search(ln):
                alc += 1
        return (ldg, stg, loc, shr, sy, fpc, inc, sfc, alc)

    def _find_block_loop_membership(self, loop_iters: Dict[int,int]) -> Dict[int,Set[int]]:
        block_map = defaultdict(set)
        for head in loop_iters.keys():
            visited=set()
            queue=deque([head])
            while queue:
                c=queue.popleft()
                if c in visited:
                    continue
                visited.add(c)
                block_map[c].add(head)
                for nxt in self.cfg[c]:
                    if nxt==head:
                        continue
                    queue.append(nxt)
        return block_map

    def _coalescing_breakdown(self, effective_block_x: int, global_ops: float) -> Tuple[float,float,float]:
        if global_ops <= 0:
            return (0.0, 0.0, 0.0)

        warp_size = self.arch.attrs.get('WARP_SIZE', 32)
        partial_slope = float(self.calib.get("partial_coalesce_slope", 0.0))
        partial_int = float(self.calib.get("partial_coalesce_intercept", 1.0))

        fullWarpsX = effective_block_x // warp_size
        partialWarpX = 1 if (effective_block_x % warp_size != 0) else 0
        totalWarpsX = fullWarpsX + partialWarpX
        if totalWarpsX <= 0:
            return (0, global_ops, 0)

        fractionFullyCoalesced = float(fullWarpsX) / totalWarpsX
        fractionPartial = float(partialWarpX) / totalWarpsX

        # now use the regression to define how "effective" partial warps are:
        # for example, partial_coalesce_factor = partial_slope * fractionPartial + partial_int
        partial_coalesce_factor = partial_slope * fractionPartial + partial_int
        # clamp 0..1 if needed
        if partial_coalesce_factor > 1.0:
            partial_coalesce_factor = 1.0
        elif partial_coalesce_factor < 0.0:
            partial_coalesce_factor = 0.0

        # total coalescing fraction
        cFrac = fractionFullyCoalesced + partial_coalesce_factor*fractionPartial
        if cFrac > 1.0:
            cFrac = 1.0
        uFrac = 1.0 - cFrac
        mem_coal = global_ops * cFrac
        mem_un = global_ops * uFrac
        mem_part = 0.0  # or you can store partial if you prefer
        return (mem_coal, mem_un, mem_part)
