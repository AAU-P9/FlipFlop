#!/usr/bin/env python3
import re
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Set, Optional, Any
from lark import Lark, Transformer, Token, v_args

from lark import Lark, Transformer, Token, v_args

PTX_GRAMMAR = r"""
start: statement*

// A statement can be a directive, instruction, or label, each optionally followed by ";"
statement: directive [";"] 
         | instruction [";"] 
         | label

// Directives begin with a DOT, then have a directive name (which can have multiple dots), plus optional operands.
directive: DOT DIRECTIVE_NAME [operand ("," operand)*]

// An instruction may be optionally predicated (e.g., "@p0") and then an opcode, which can itself have multiple dotted qualifiers.
instruction: [predicate] OPCODE [operand ("," operand)*]

// A label is just an identifier followed by a colon.
label: CNAME ":"

// --- Terminals ---

// We allow multiple '.' segments in both directives and opcodes. For example:
//    .visible .entry
//    ld.global.u32
DIRECTIVE_NAME: /[a-zA-Z_$][\w$.]*/
OPCODE:         /[a-zA-Z_$][\w$.]*/

// A "CNAME" is used for generic identifiers such as register names, label names, etc.
CNAME: /[a-zA-Z_$][\w$]*/

// A DOT token for convenience.
DOT: "."

// Operands can be registers, immediates, addresses, predicates, or generic CNAMEs.
operand: register
        | immediate
        | address
        | predicate
        | CNAME

// A register starts with "%" followed by a CNAME.
register: "%" CNAME

// An immediate can be a plain number or that same number in parentheses.
immediate: number
          | "(" number ")"

// A memory address: either a register (optionally with an offset) or a number inside brackets.
address: "[" (register [("+" | "-") number] | number)? "]"

// A predicate is simply "@" followed by a CNAME.
predicate: "@" CNAME

// A number can be:
//    - an integer (possibly negative), e.g. -123
//    - a float with optional exponent, e.g. 3.14e+1
//    - or a hexadecimal literal, e.g. 0xdeadbeef
number: /-?\d+(\.\d+)?([eE][+-]?\d+)?/ | /0x[0-9A-Fa-f]+/

// Optional semicolon is recognized in the grammar but not strictly required.
SEMI: ";"

// Ignore whitespace
%ignore /\s+/

// Ignore both line (// ...) and block (/* ... */) comments
%ignore /\/\/[^\n]*/
%ignore /\/\*[^*]*\*+([^/*][^*]*\*+)*\//
"""


@v_args(inline=True)
class PTXTransformer(Transformer):
    def start(self, *statements):
        return {"type": "program", "statements": list(statements)}

    def statement(self, stmt):
        # Each statement is either a directive, instruction, or label. 
        # We'll just pass it through to keep the AST simpler.
        return stmt

    def directive(self, dot, name, *operands):
        return {
            "type": "directive",
            "name": name,  # e.g., "visible", "entry", "param"
            "operands": list(operands)
        }

    def instruction(self, maybe_pred, opcode, *operands):
        # The grammar says [predicate], so maybe_pred is either a dict for the predicate or None.
        pred = None
        actual_opcode = opcode
        # If the first argument is not a Token but a dict with type="predicate", that means the instruction was predicated.
        if isinstance(maybe_pred, dict) and maybe_pred.get("type") == "predicate":
            pred = maybe_pred
            actual_opcode = opcode
        else:
            # No predicate was given
            pred = None
            # shift the opcode left
            actual_opcode = maybe_pred
            operands = (opcode,) + operands

        return {
            "type": "instruction",
            "predicate": pred,      # e.g., {"type": "predicate", "name": "p0"} or None
            "opcode": actual_opcode, # e.g. "ld.global.u32", "bra", "add.cc", etc.
            "operands": list(operands)
        }

    def label(self, name):
        return {"type": "label", "name": name}

    def register(self, percent, name):
        return {"type": "register", "name": name}

    def immediate(self, value):
        # If value is a float or int token, we already parse it in `number(...)`.
        # If it's parentheses, it's a nested parse call. We can just return it.
        return {"type": "immediate", "value": value}

    def address(self, lbrack, *args):
        # `args` can be empty or contain up to three items: (register, op, number)
        base = None
        offset = 0
        if args:
            # If the first is a register dict, use it as the base.
            if isinstance(args[0], dict) and args[0].get("type") == "register":
                base = args[0]
                if len(args) >= 3:
                    # Expect something like ('+', number) or ('-', number)
                    operator = args[1]
                    number_val = args[2]
                    offset = number_val if operator == "+" else -number_val
            else:
                # The bracket had just a numeric immediate
                offset = args[0]
        return {"type": "address", "base": base, "offset": offset}

    def predicate(self, at, name):
        return {"type": "predicate", "name": name}

    def number(self, token):
        # Convert to numeric. If it's hex, handle that separately.
        val = token.value
        if val.lower().startswith("0x"):
            return int(val, 16)
        else:
            # parse float if it has '.', 'e', or 'E', else parse int
            if any(c in val for c in ['.', 'e', 'E']):
                return float(val)
            else:
                return int(val)

    def OPCODE(self, token):
        return token.value

    def DIRECTIVE_NAME(self, token):
        return token.value

    def CNAME(self, token):
        return token.value


# --- PTX Parser Wrapper ---
class PTXParser:
    def __init__(self):
        self.parser = Lark(PTX_GRAMMAR, parser="lalr", transformer=PTXTransformer())
        
    def parse(self, code: str) -> Any:
        try:
            return self.parser.parse(code)
        except Exception as e:
            print(f"Error parsing PTX code: {e}")
            return None

# --- (DependencyAnalyzer and other analysis classes remain as you have them) ---

# --- Enhanced PTX Analyzer using the grammar ---
class EnhancedPTXAnalyzer:
    def __init__(self, ptx_code: str, block_dims: Tuple[int, int]):
        self.parser = PTXParser()
        self.ast = self.parser.parse(ptx_code)
        if self.ast is None:
            raise ValueError("Failed to parse PTX code.")
        self.block_x, self.block_y = block_dims
        self.instructions = []
        self.labels = {}
        self.dependency_graph = defaultdict(list)
        self.memory_ops = []
        self._preprocess_ast()
        self._analyze_dependencies()
        self._analyze_memory_patterns()

    def _preprocess_ast(self):
        # Flatten the AST assuming top-level 'program' contains a list of statements.
        for stmt in self.ast.get("statements", []):
            if stmt.get("type") in {"instruction", "label", "directive"}:
                self.instructions.append(stmt)
                if stmt.get("type") == "label":
                    self.labels[stmt.get("name")] = len(self.instructions) - 1

    def _analyze_dependencies(self):
        analyzer = DependencyAnalyzer(self.instructions)
        self.dependency_graph = analyzer.analyze()
        self.critical_path = analyzer.get_critical_path()

    def _analyze_memory_patterns(self):
        warp_size = 32  # Assume warp size of 32
        coalescing_threshold = warp_size * 4  # For example, 128 bytes per memory transaction
        for instr in self.instructions:
            opcode = instr.get("opcode", "").lower()
            if opcode.startswith("ld.global") or opcode.startswith("st.global"):
                for op in instr.get("operands", []):
                    if isinstance(op, dict) and op.get("type") == "address":
                        offset = op.get("offset", 0)
                        stride = offset * 4  # Assume each offset unit corresponds to 4 bytes
                        coalesced = abs(stride) <= coalescing_threshold
                        self.memory_ops.append({
                            "opcode": opcode,
                            "stride": stride,
                            "coalesced": coalesced
                        })

    def get_coalescing_efficiency(self) -> float:
        if not self.memory_ops:
            return 1.0
        coalesced = sum(1 for op in self.memory_ops if op["coalesced"])
        return coalesced / len(self.memory_ops)

    def get_instruction_mix(self) -> Dict[str, int]:
        mix = defaultdict(int)
        for instr in self.instructions:
            if instr.get("type") != "instruction":
                continue
            opcode = instr.get("opcode", "").lower()
            if opcode.startswith("ld.global") or opcode.startswith("st.global"):
                mix["global_mem"] += 1
            elif opcode.startswith("ld.local") or opcode.startswith("st.local"):
                mix["local_mem"] += 1
            elif opcode.startswith("ld.shared") or opcode.startswith("st.shared"):
                mix["shared_mem"] += 1
            elif opcode.startswith("bar.sync"):
                mix["sync"] += 1
            elif any(opcode.startswith(p) for p in ["fadd", "fsub", "fmul", "fma", "fdiv"]):
                mix["fp"] += 1
            elif opcode in {"add", "sub", "mul", "div", "rem", "min", "max"}:
                mix["int"] += 1
            elif any(opcode.startswith(p) for p in ["sin", "cos", "sqrt", "rsqrt", "rcp"]):
                mix["sfu"] += 1
            elif opcode.startswith("bra") or opcode in {"ret", "exit"}:
                mix["control"] += 1
            else:
                mix["other"] += 1
        return mix

# --- Example usage ---
if __name__ == "__main__":
    sample_ptx = r"""
    //
        // Generated by NVIDIA NVVM Compiler
        //
        // Compiler Build ID: CL-33961263
        // Cuda compilation tools, release 12.4, V12.4.99
        // Based on NVVM 7.0.1
        //

        .version 8.4
        .target sm_89
        .address_size 64

            // .globl	matMul

        .visible .entry matMul(
            .param .u64 matMul_param_0,
            .param .u64 matMul_param_1,
            .param .u64 matMul_param_2,
            .param .u32 matMul_param_3,
            .param .u32 matMul_param_4,
            .param .u32 matMul_param_5
        )
        {
            .reg .pred 	%p<9>;
            .reg .f32 	%f<30>;
            .reg .b32 	%r<32>;
            .reg .b64 	%rd<34>;


            ld.param.u64 	%rd18, [matMul_param_0];
            ld.param.u64 	%rd19, [matMul_param_1];
            ld.param.u64 	%rd17, [matMul_param_2];
            ld.param.u32 	%r14, [matMul_param_3];
            ld.param.u32 	%r12, [matMul_param_4];
            ld.param.u32 	%r13, [matMul_param_5];
            cvta.to.global.u64 	%rd1, %rd19;
            cvta.to.global.u64 	%rd2, %rd18;
            mov.u32 	%r15, %ntid.y;
            mov.u32 	%r16, %ctaid.y;
            mov.u32 	%r17, %tid.y;
            mad.lo.s32 	%r1, %r16, %r15, %r17;
            mov.u32 	%r18, %ntid.x;
            mov.u32 	%r19, %ctaid.x;
            mov.u32 	%r20, %tid.x;
            mad.lo.s32 	%r2, %r19, %r18, %r20;
            setp.ge.s32 	%p1, %r1, %r14;
            setp.ge.s32 	%p2, %r2, %r13;
            or.pred  	%p3, %p1, %p2;
            @%p3 bra 	$L__BB0_9;

            setp.lt.s32 	%p4, %r12, 1;
            mov.f32 	%f29, 0f00000000;
            @%p4 bra 	$L__BB0_8;

            add.s32 	%r22, %r12, -1;
            and.b32  	%r31, %r12, 3;
            setp.lt.u32 	%p5, %r22, 3;
            mov.f32 	%f29, 0f00000000;
            mov.u32 	%r30, 0;
            @%p5 bra 	$L__BB0_5;

            sub.s32 	%r29, %r12, %r31;
            mul.lo.s32 	%r24, %r12, %r1;
            mul.wide.s32 	%rd3, %r24, 4;
            mul.wide.s32 	%rd20, %r2, 4;
            add.s64 	%rd30, %rd1, %rd20;
            mul.wide.s32 	%rd5, %r13, 4;
            mov.f32 	%f29, 0f00000000;
            mov.u32 	%r30, 0;
            mov.u64 	%rd31, %rd2;

        $L__BB0_4:
            add.s64 	%rd21, %rd31, %rd3;
            ld.global.f32 	%f12, [%rd30];
            ld.global.f32 	%f13, [%rd21];
            fma.rn.f32 	%f14, %f13, %f12, %f29;
            add.s64 	%rd22, %rd30, %rd5;
            ld.global.f32 	%f15, [%rd22];
            ld.global.f32 	%f16, [%rd21+4];
            fma.rn.f32 	%f17, %f16, %f15, %f14;
            add.s64 	%rd23, %rd22, %rd5;
            ld.global.f32 	%f18, [%rd23];
            ld.global.f32 	%f19, [%rd21+8];
            fma.rn.f32 	%f20, %f19, %f18, %f17;
            add.s64 	%rd24, %rd23, %rd5;
            add.s64 	%rd30, %rd24, %rd5;
            ld.global.f32 	%f21, [%rd24];
            ld.global.f32 	%f22, [%rd21+12];
            fma.rn.f32 	%f29, %f22, %f21, %f20;
            add.s32 	%r30, %r30, 4;
            add.s64 	%rd31, %rd31, 16;
            add.s32 	%r29, %r29, -4;
            setp.ne.s32 	%p6, %r29, 0;
            @%p6 bra 	$L__BB0_4;

        $L__BB0_5:
            setp.eq.s32 	%p7, %r31, 0;
            @%p7 bra 	$L__BB0_8;

            mad.lo.s32 	%r25, %r30, %r13, %r2;
            mul.wide.s32 	%rd25, %r25, 4;
            add.s64 	%rd33, %rd1, %rd25;
            mul.wide.s32 	%rd11, %r13, 4;
            mad.lo.s32 	%r26, %r12, %r1, %r30;
            mul.wide.s32 	%rd26, %r26, 4;
            add.s64 	%rd32, %rd2, %rd26;

        $L__BB0_7:
            .pragma "nounroll";
            ld.global.f32 	%f23, [%rd33];
            ld.global.f32 	%f24, [%rd32];
            fma.rn.f32 	%f29, %f24, %f23, %f29;
            add.s64 	%rd33, %rd33, %rd11;
            add.s64 	%rd32, %rd32, 4;
            add.s32 	%r31, %r31, -1;
            setp.ne.s32 	%p8, %r31, 0;
            @%p8 bra 	$L__BB0_7;

        $L__BB0_8:
            mad.lo.s32 	%r27, %r1, %r13, %r2;
            cvta.to.global.u64 	%rd27, %rd17;
            mul.wide.s32 	%rd28, %r27, 4;
            add.s64 	%rd29, %rd27, %rd28;
            st.global.f32 	[%rd29], %f29;

        $L__BB0_9:
            ret;

        }
    """
    try:
        analyzer = EnhancedPTXAnalyzer(sample_ptx, (256, 1))
        print("Coalescing Efficiency:", analyzer.get_coalescing_efficiency())
        print("Instruction Mix:", analyzer.get_instruction_mix())
        print("Critical Path Length:", len(analyzer.critical_path))
    except Exception as e:
        print("An error occurred during analysis:", e)
