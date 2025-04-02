#!/usr/bin/env python3
"""
Enhanced PTX Parser and Analyzer

This script provides:
1. A high-coverage Lark grammar for modern PTX.
2. A PTXTransformer that constructs an AST from parsed PTX.
3. A PTXAnalyzer class that:
   - Analyzes directives for register/shared usage.
   - Classifies instructions (including memory space).
   - Tracks dependencies and builds a dependency graph.
   - Estimates critical path and preliminary instruction latencies.
   - Returns a KernelAnalysis object summarizing kernel characteristics.

Author: [Your Name]
"""

import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple, Union
from lark import Lark, Transformer, v_args, Token

# ============================================
#   PTX Grammar (Lark)
#   Covers modern PTX constructs with expanded builtin types.
# ============================================
PTX_GRAMMAR = r"""
start: (label | kernel | directive | instruction | _NL)*

kernel: ".entry" cname ["(" params ")"] "{" statement* "}"
      | ".func" cname ["(" params ")"] "{" statement* "}"
params: param ("," param)*

param: ".param" type_ (".align" dec_lit)? cname ( "[" dec_lit "]" )?

directive: "." directive_name (operand | modifier)* [(";")]?

instruction: [predicate] opcode operand_list? ("," predicate_guard)? [(";")]?

operand_list: operand ("," operand)*

label: cname ":" _NL?

_NL: /(\r?\n)+/

// Statements within a kernel body
statement: label
         | directive
         | instruction
         | _NL

// Operands
operand: register
       | immediate
       | address
       | param_ref
       | string_lit
       | cname
       | builtin_var
       | pack_expression

// Pack or composite operand forms (e.g. { %r1, %r2 })
pack_expression: "{" [operand ("," operand)*] "}"

builtin_var: "%" (env_reg | special_var)
env_reg: /ctaid\.[xyz]/ | /ntid\.[xyz]/ | /tid\.[xyz]/ | "warpid" | "laneid" | "nctaid" | "nsmid" | "smid" | "gridid"
special_var: /[\w\.\%\$\_]+/

register: "%" cname group_spec?
group_spec: "<" dec_lit ">"

address: "[" [base (("+"|"-") offset_expr)?] "]"
base: register | dec_lit | hex_lit | cname
offset_expr: (register | number)
           | (number ("+"|"-") number)

immediate: number
         | float_lit
         | "generic" | "volatile" | "const" | "param" | "global" | "local"

param_ref: "param" cname
predicate: "@" cname
predicate_guard: "{" cname "}"

type_: builtin_type
      | vector_type
      | array_type
      | user_type

// Modified builtin_type allows an optional leading period
builtin_type: /(\.?[su](8|16|32|64)|\.?b8|\.?b16|\.?b32|\.?b64|\.?f16|\.?f32|\.?f64|\.?pred)/

vector_type: ".v" dec_lit "." builtin_type
array_type: builtin_type "[" dec_lit "]"
user_type: cname

modifier: "." modifier_name
modifier_name: /[a-zA-Z0-9_]+/

directive_name: /[a-zA-Z_]\w*(\.[\w\.]+)*/

%import common.ESCAPED_STRING
%import common.WS
%ignore WS

opcode: /(ld|st|mov|add|sub|mul|div|bra|ret|bar|shl|shr|and|or|xor|fadd|fsub|fmul|fdiv|cvt|sin|cos|sqrt|rsqrt|rcp|min|max|neg|abs|mad|atom|red|vote|cp|bmma|mma|prmt|selp|setp|slct|warp|shfl|activemask)\.[a-z0-9_\.]*/i
       | /[a-z]+/i

cname: /[a-zA-Z_$][\w$\.]*/

float_lit: /-?\d+\.\d+([eE][+-]?\d+)?/
dec_lit: /-?\d+/
hex_lit: /0x[0-9a-fA-F]+/
string_lit: /"[^"]*"/

number: dec_lit | hex_lit

%ignore /\/\/.*/
%ignore /\/\*.*?\*\//s
%ignore /[ \t\r\n\f]+/
%ignore "//" /[^\n]*/
%ignore "//" /.*/
"""

# ============================================
#   AST Transformer
# ============================================
@v_args(inline=True)
class PTXTransformer(Transformer):
    def __init__(self):
        super().__init__()
        self.current_kernel = None

    def start(self, *items):
    # Filter out None items (e.g. from _NL)
        return {"type": "program", "statements": [item for item in items if item is not None]}


    def kernel(self, entry_kw, name, params, *body):
        return {
            "type": "kernel",
            "name": name,
            "params": params.children if hasattr(params, 'children') else [],
            "body": [b for b in body if b is not None]
        }

    def params(self, *args):
        return list(args)

    def param(self, *args):
        items = list(args)
        param_dict = {"type": "param", "ptype": None, "align": None, "name": None, "array_size": None}
        for it in items:
            if isinstance(it, dict) and it.get("type") == "type":
                param_dict["ptype"] = it["name"]
            elif isinstance(it, int):
                if param_dict["align"] is None:
                    param_dict["align"] = it
                else:
                    param_dict["array_size"] = it
            elif isinstance(it, str):
                if param_dict["name"] is None:
                    param_dict["name"] = it
        return param_dict

    def directive(self, dot, name, *parts):
        # Extract name from Token or Tree
        if isinstance(name, Token):
            name_val = name.value
        elif hasattr(name, 'children') and isinstance(name.children[0], Token):
            name_val = name.children[0].value
        else:
            name_val = str(name)
        
        operands = []
        modifiers = []
        for part in parts:
            if isinstance(part, dict) and 'type' in part:
                operands.append(part)
            else:
                modifiers.append(part)
        return {
            "type": "directive",
            "name": name_val,
            "modifiers": modifiers,
            "operands": operands
        }


    def instruction(self, *parts):
        predicate = None
        opcode = None
        modifiers = []
        operands = []
        predicate_guard = None
        for part in parts:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype == "predicate":
                    predicate = part
                elif ptype == "predicate_guard":
                    predicate_guard = part
                else:
                    operands.append(part)
            elif isinstance(part, Token):
                if part.type == "OPCODE":
                    opcode = part.value
                else:
                    modifiers.append(part.value)
            else:
                if isinstance(part, str):
                    if opcode is None:
                        opcode = part
                    else:
                        modifiers.append(part)
                else:
                    operands.append(part)
        return {
            "type": "instruction",
            "predicate": predicate,
            "opcode": (opcode.lower() if opcode else None),
            "modifiers": modifiers,
            "operands": operands,
            "predicate_guard": predicate_guard
        }

    def label(self, lbl):
        return {"type": "label", "name": lbl}

    def statement(self, item):
        # Return an empty statement_list if item is None
        if item is None:
            return {"type": "statement_list", "statements": []}
        return {"type": "statement_list", "statements": [item]}
    
    def _NL(self, _):
        return None  # Ignore newlines in AST

    def directive_name(self, token):
        return token.value

    def cname(self, token):
        return token.value

    def address(self, *components):
        if len(components) == 0:
            return {"type": "address", "base": None, "offset": None}
        if len(components) == 1:
            return {"type": "address", "base": components[0], "offset": None}
        base_part = components[0]
        offset_part = components[1]
        return {"type": "address", "base": base_part, "offset": offset_part}

    def base(self, val):
        return val

    def offset_expr(self, val):
        return val

    def register(self, cname, group_spec=None):
        return {"type": "register", "name": cname, "group": group_spec}

    def immediate(self, val):
        return {"type": "immediate", "value": self._parse_number(val)}

    def builtin_var(self, val):
        return {"type": "builtin_var", "name": val}

    def pack_expression(self, *ops):
        return {"type": "pack_expression", "operands": list(ops)}

    def param_ref(self, *items):
        return {"type": "param_ref", "name": items[-1]}

    def predicate(self, pred_name):
        return {"type": "predicate", "name": pred_name}

    def predicate_guard(self, pred_name):
        return {"type": "predicate_guard", "name": pred_name}

    def type_(self, t):
        return {"type": "type", "name": t}

    def DEC_LIT(self, token):
        return int(token.value)

    def HEX_LIT(self, token):
        return int(token.value, 16)

    def FLOAT_LIT(self, token):
        return float(token.value)

    def number(self, val):
        if isinstance(val, (float, int)):
            return val
        return self._parse_number(val)

    def _parse_number(self, s) -> Union[int, float]:
        if isinstance(s, Token):
            s = s.value
        if isinstance(s, str):
            s = s.strip()
            if s.startswith("0x"):
                return int(s, 16)
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        return s

# ============================================
#   Architecture Configuration
# ============================================
class ArchitectureConfig:
    def __init__(self,
                 warp_size=32,
                 max_threads_per_block=1024,
                 shared_mem_size=49152,
                 register_file_size=65536,
                 clock_rate_ghz=1.0,
                 sm_count=80):
        self.warp_size = warp_size
        self.max_threads_per_block = max_threads_per_block
        self.shared_mem_size = shared_mem_size
        self.register_file_size = register_file_size
        self.clock_rate_ghz = clock_rate_ghz
        self.sm_count = sm_count

# ============================================
#   Kernel Analysis
# ============================================
class KernelAnalysis:
    def __init__(self):
        self.inst_counts = defaultdict(int)
        self.mem_access = defaultdict(int)
        self.registers_used = 0
        self.shared_mem_used = 0
        self.dependencies = []
        self.critical_path = 0
        self.control_flow = []
        self.register_usage_map = {}
        self.inst_latency = 0

    def __repr__(self):
        return (f"KernelAnalysis("
                f"inst_counts={dict(self.inst_counts)}, "
                f"mem_access={dict(self.mem_access)}, "
                f"registers_used={self.registers_used}, "
                f"shared_mem_used={self.shared_mem_used}, "
                f"critical_path={self.critical_path}, "
                f"dependencies={len(self.dependencies)})")

# ============================================
#   Dependency Graph
# ============================================
class DependencyGraph:
    def __init__(self):
        self.nodes = []
        self.edges = defaultdict(list)
        self.latencies = {}
        self.last_writer_of_reg = {}

    def add_instruction(self, instr: dict, latency: int) -> int:
        idx = len(self.nodes)
        self.nodes.append(instr)
        self.latencies[idx] = latency
        return idx

    def add_dependency(self, src: int, dest: int):
        self.edges[src].append(dest)

    def critical_path_length(self) -> int:
        dist = [0] * len(self.nodes)
        for u in range(len(self.nodes)):
            for v in self.edges[u]:
                cand = dist[u] + self.latencies[v]
                if cand > dist[v]:
                    dist[v] = cand
        return max(dist) if dist else 0

# ============================================
#   Enhanced PTX Analyzer
# ============================================
class PTXAnalyzer:
    _INSTRUCTION_LATENCY = {
        "ld": 100, "st": 100, "fadd": 4, "fmul": 5, "ffma": 5,
        "iadd": 1, "imul": 3, "mov": 1, "bra": 2, "bar": 10,
        "cp.async": 50, "cp.wait": 10, "bmma.sync": 20, "mma.sync": 20,
    }

    _MEMORY_SPACES = {"global", "shared", "local", "const", "param", "tex"}

    def __init__(self, ptx_code: str, arch_config: ArchitectureConfig):
        self.ptx_code = ptx_code
        self.arch = arch_config
        self.parser = Lark(
            PTX_GRAMMAR,
            parser='lalr',
            lexer='contextual',
            transformer=PTXTransformer()
        )
        self.analysis = KernelAnalysis()
        self.dependency_graph = DependencyGraph()

    def analyze(self) -> KernelAnalysis:
        try:
            parsed_ast = self.parser.parse(self.ptx_code)
            # Print the parsed AST for debugging purposes
            print("Parsed AST:")
            print(parsed_ast)
            self._traverse_ast_and_analyze(parsed_ast)
            self.analysis.critical_path = self.dependency_graph.critical_path_length()
        except Exception as e:
            print(f"[PTXAnalyzer] Error while analyzing PTX: {e}")
        return self.analysis

    def _traverse_ast_and_analyze(self, ast_node):
        if ast_node is None:
            return
        if not isinstance(ast_node, dict):
            return
        nodetype = ast_node.get("type", None)
        if nodetype == "program":
            for stmt in ast_node.get("statements", []):
                if stmt is not None:
                    self._traverse_ast_and_analyze(stmt)
        elif nodetype == "kernel":
            body = ast_node.get("body", [])
            for stmt in body:
                if stmt is not None:
                    self._analyze_statement(stmt)
        else:
            if nodetype in ["directive", "instruction", "label", "statement_list"]:
                self._analyze_statement(ast_node)

    def _analyze_statement(self, stmt):
        stype = stmt.get("type")
        if stype == "statement_list":
            for s in stmt.get("statements", []):
                self._analyze_statement(s)
        elif stype == "directive":
            self._process_directive(stmt)
        elif stype == "instruction":
            self._process_instruction(stmt)
        elif stype == "label":
            self.analysis.control_flow.append(stmt["name"])

    def _process_directive(self, directive):
        name = directive["name"].lower()
        if name == ".reg":
            for op in directive.get("operands", []):
                if isinstance(op, dict) and op.get("type") == "register":
                    if "group" in op and isinstance(op["group"], int):
                        self.analysis.registers_used += op["group"]
                    else:
                        self.analysis.registers_used += 1
        elif name == ".shared":
            for op in directive.get("operands", []):
                if isinstance(op, dict) and op.get("type") == "address":
                    if isinstance(op.get("offset"), int):
                        self.analysis.shared_mem_used += op["offset"]

    def _directive_to_text(self, directive) -> str:
        parts = []
        # If the "name" field is None, use an empty string.
        if directive.get("name") is not None:
            parts.append(directive["name"])
        for m in directive.get("modifiers", []):
            if m is not None:
                parts.append(str(m))
        for op in directive.get("operands", []):
            if op is not None:
                if isinstance(op, dict):
                    parts.append(op.get("name", ""))
                else:
                    parts.append(str(op))
        return " ".join(parts)

    def _process_instruction(self, instr):
        opcode = instr["opcode"] if instr["opcode"] else "unknown"
        base_op = opcode.split('.')[0] if opcode else "unknown"
        self.analysis.inst_counts[base_op] += 1
        for space in self._MEMORY_SPACES:
            if space in opcode:
                self.analysis.mem_access[space] += 1
        lat = self._get_latency(opcode)
        idx = self.dependency_graph.add_instruction(instr, lat)
        writes, reads = self._get_def_use_sets(instr)
        for r in reads:
            if r in self.dependency_graph.last_writer_of_reg:
                writer_id = self.dependency_graph.last_writer_of_reg[r]
                self.dependency_graph.add_dependency(writer_id, idx)
        for w in writes:
            self.dependency_graph.last_writer_of_reg[w] = idx

    def _get_latency(self, opcode: str) -> int:
        base = opcode.split('.')[0].lower()
        return self._INSTRUCTION_LATENCY.get(base, 
            100 if base in ['ld', 'st'] else 
            10 if base == 'bar' else 
            4  # Default ALU latency
        )

    def _get_def_use_sets(self, instr: dict) -> Tuple[List[str], List[str]]:
        writes = []
        reads = []
        
        # Handle 3-operand forms
        if len(ops) >= 3 and base_op in ['mad', 'fma']:
            if _is_reg(ops[0]): writes.append(ops[0]['name'])
            if _is_reg(ops[1]): reads.append(ops[1]['name'])
            if _is_reg(ops[2]): reads.append(ops[2]['name'])
            if len(ops) > 3 and _is_reg(ops[3]): reads.append(ops[3]['name'])
        
        # Handle memory operations
        elif base_op.startswith(('ld', 'st')):
            for op in ops:
                if op.get('type') == 'address':
                    if _is_reg(op.get('base')): reads.append(op['base']['name'])
                    if isinstance(op.get('offset'), dict) and _is_reg(op['offset']):
                        reads.append(op['offset']['name'])
        
        # Handle generic case
        else:
            if ops and _is_reg(ops[0]): 
                writes.append(ops[0]['name'])
            for op in ops[1:]:
                if _is_reg(op): reads.append(op['name'])
        
        return writes, reads


# ============================================
#   Example usage
# ============================================
def main():
    sample_ptx = r"""
    .version 8.4
    .target sm_90
    .address_size 64

    .visible .entry example_kernel(
        .param .u64 ptr_a,
        .param .u64 ptr_b,
        .param .u64 ptr_c
    )
    {
        .reg .u64 %r<3>;
        .reg .f32 %f<4>;
        .shared .align 4 .b8 smem[128];

    L0:
        ld.param.u64 %r0, [ptr_a];
        ld.param.u64 %r1, [ptr_b];
        ld.param.u64 %r2, [ptr_c];

        mov.u32 %f0, %tid.x;
        shl.b32 %f1, %f0, 2;

        ld.global.f32 %f2, [%r0 + %f1];
        ld.global.f32 %f3, [%r1 + %f1];

        add.f32 %f2, %f2, %f3;
        st.global.f32 [%r2 + %f1], %f2;

        bra L2;
    L2:
        ret;
    }
    """

    arch_config = ArchitectureConfig(
        warp_size=32,
        max_threads_per_block=1024,
        shared_mem_size=49152,
        register_file_size=65536,
        clock_rate_ghz=1.4,
        sm_count=80
    )

    analyzer = PTXAnalyzer(sample_ptx, arch_config)
    results = analyzer.analyze()
    print(results)

if __name__ == "__main__":
    main()
