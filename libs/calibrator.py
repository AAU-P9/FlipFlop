import csv
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import time

from cubindings import execute_code
from calibrator.cubindings_analyser import run_analyser

# ============================================================================
# Configuration
# ============================================================================
GRID = 152          # 2 blocks per SM on AD103 (76 SMs)
BLOCK = 1024
TARGET_NS = 5_000_000_000  # 5 seconds
PILOT_ITERS = 100_000
WEIGHTS_PATH = "/home/p10/aau-p10-ptx-energy/linear-model/weights.csv" # Set to p10 since all users have access there
BUF_BYTES_PER_THREAD = 1024
PILOT_CACHE_PATH = Path(__file__).parent / "pilot_cache.json"
METRICS_WARMUP_S = 1


def _load_pilot_cache() -> dict:
    if not PILOT_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(PILOT_CACHE_PATH.read_text())
        cfg = data.get("config", {})
        if cfg.get("pilot_iters") != PILOT_ITERS or cfg.get("target_ns") != TARGET_NS:
            print("  [cache] config changed, discarding pilot cache", flush=True)
            return {}
        return data.get("results", {})
    except Exception:
        return {}


def _save_pilot_cache(results: dict) -> None:
    data = {
        "config": {"pilot_iters": PILOT_ITERS, "target_ns": TARGET_NS},
        "results": results,
    }
    PILOT_CACHE_PATH.write_text(json.dumps(data, indent=2))


def _spinner():
    while True:
        for cursor in '|/-\\':
            yield cursor

def spinning_cursor(stop_event: threading.Event):
    spinner_generator = _spinner()
    while not stop_event.is_set():
        sys.stdout.write(next(spinner_generator))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
    sys.stdout.write(' \b')
    sys.stdout.flush()

@contextmanager
def _spinner_running(label: str):
    print(f"    {label} ", end="", flush=True)
    stop = threading.Event()
    t = threading.Thread(target=spinning_cursor, args=(stop,), daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()

# ============================================================================
# Instruction templates — latency variant only (loop-carried dep on `d`)
#
# Inline asm constraints: r=.b32/.s32/.u32, l=.b64/.s64/.u64, f=.f32
# "+x" = read+write, "=x" = write-only, "x" = read-only
# ============================================================================

INSTRUCTION_TEMPLATES = {
    "add.f32": {
        "setup": "float b = 1.5f, d = (float)tid;",
        "asm":   '"add.f32 %0, %0, %1;" : "+f"(d) : "f"(b)',
        "sink":  "((float*)sink)[tid] = d;",
    },
    "add.s32": {
        "setup": "int b = 1, d = tid;",
        "asm":   '"add.s32 %0, %0, %1;" : "+r"(d) : "r"(b)',
        "sink":  "((int*)sink)[tid] = d;",
    },
    "add.s64": {
        "setup": "long long b = 1, d = tid;",
        "asm":   '"add.s64 %0, %0, %1;" : "+l"(d) : "l"(b)',
        "sink":  "((long long*)sink)[tid] = d;",
    },
    "mul.lo.s32": {
        "setup": "int a = tid | 1, d = 1;",
        "asm":   '"mul.lo.s32 %0, %0, %1;" : "+r"(d) : "r"(a)',
        "sink":  "((int*)sink)[tid] = d;",
    },
    "mov.b32": {
        "setup": "int tmp = tid;",
        "asm":   '"mov.b32 %0, %0;" : "+r"(tmp)',
        "sink":  "((int*)sink)[tid] = tmp;",
    },
    "mov.u32": {
        "setup": "unsigned tmp = tid;",
        "asm":   '"mov.u32 %0, %0;" : "+r"(tmp)',
        "sink":  "((unsigned*)sink)[tid] = tmp;",
    },
    "cvt.s64.s32": {
        "setup": "int a = tid; long long d = 0;",
        "asm":   ('"cvt.s64.s32 %0, %1;\\n\\t"'
                  '"cvt.s32.s64 %1, %0;"'
                  ' : "+l"(d), "+r"(a)'),
        "sink":  "((long long*)sink)[tid] = d;",
    },
    "shl.b64": {
        "setup": "long long d = tid | 1; unsigned s = 1;",
        "asm":   '"shl.b64 %0, %0, %1;" : "+l"(d) : "r"(s)',
        "sink":  "((long long*)sink)[tid] = d;",
    },
    "setp.lt.s32": {
        "setup": "int b = 512, d = 0;",
        "asm":   ('".reg .pred %%p%=;\\n\\t"'
                  '"setp.lt.s32 %%p%=, %0, %1;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%p%=;"'
                  ' : "+r"(d) : "r"(b)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    "not.pred": {
        "setup": "int a = tid & 1, d = 0;",
        "asm":   ('".reg .pred %%p%=, %%q%=;\\n\\t"'
                  '"setp.ne.s32 %%p%=, %1, 0;\\n\\t"'
                  '"not.pred %%q%=, %%p%=;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%q%=;"'
                  ' : "=r"(d) : "r"(a)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    "ld.f32": {
        "setup":     "float d = 0.0f; float* p = buf + tid;",
        "asm":       '"ld.f32 %0, [%1];" : "=f"(d) : "l"(p)',
        "sink":      "((float*)sink)[tid] = d;",
        "needs_buf": True,
    },
    "st.f32": {
        "setup":     "float v = (float)tid; float* p = buf + tid;",
        "asm":       '"st.f32 [%0], %1;" :: "l"(p), "f"(v)',
        "sink":      "",
        "needs_buf": True,
    },
    "mul.f32": {
        "setup": "float b = 1.0f, d = (float)(tid + 1);",
        "asm":   '"mul.f32 %0, %0, %1;" : "+f"(d) : "f"(b)',
        "sink":  "((float*)sink)[tid] = d;",
    },
    "mov.f32": {
        "setup": "float d = (float)tid;",
        "asm":   '"mov.f32 %0, %0;" : "+f"(d)',
        "sink":  "((float*)sink)[tid] = d;",
    },
    "mov.pred": {
        "setup": "int d = tid & 1;",
        "asm":   ('".reg .pred %%p%=, %%q%=;\\n\\t"'
                  '"setp.ne.s32 %%p%=, %0, 0;\\n\\t"'
                  '"mov.pred %%q%=, %%p%=;\\n\\t"'
                  '"selp.s32 %0, 1, 0, %%q%=;"'
                  ' : "+r"(d)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    # %=  expands to a unique integer per asm instance so repeat=2 gets distinct labels
    "bra": {
        "setup": "int d = tid;",
        "asm":   ('"bra $L%=_skip;\\n\\t"'
                  '"add.s32 %0, %0, -1;\\n\\t"'
                  '"$L%=_skip:\\n\\t"'
                  '"add.s32 %0, %0, 1;"'
                  ' : "+r"(d)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
    "bra.uni": {
        "setup": "int d = tid;",
        "asm":   ('"bra.uni $L%=_skip;\\n\\t"'
                  '"add.s32 %0, %0, -1;\\n\\t"'
                  '"$L%=_skip:\\n\\t"'
                  '"add.s32 %0, %0, 1;"'
                  ' : "+r"(d)'),
        "sink":  "((int*)sink)[tid] = d;",
    },
}


# ============================================================================
# Kernel template — REPEAT copies of the asm block per loop iter
# ============================================================================

KERNEL_TEMPLATE = """
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "ptx_meta.h"
#define ITERATIONS {iters}

__global__ void ptx_kernel(void* __restrict__ sink{extra_args}) {{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    {setup}
    META_LOOP(main_loop, ITERATIONS, ITERATIONS, false);
    #pragma unroll 1
    for (int i = 0; i < ITERATIONS; ++i) {{
{asm_blocks}
    }}
    {sink}
}}

int main() {{
    void* sink;
    cudaMalloc(&sink, sizeof(long long) * {grid} * {block});
    {host_setup}
    ptx_kernel<<<{grid}, {block}>>>(sink{launch_extra});
    cudaDeviceSynchronize();
    cudaFree(sink);
    {host_teardown}
    return 0;
}}
"""


def build_program(insn, iters, repeat=1, grid=GRID, block=BLOCK):
    spec = INSTRUCTION_TEMPLATES[insn]
    needs_buf = spec.get("needs_buf", False)

    one_block = f"        asm volatile (\n            {spec['asm']}\n        );"
    asm_blocks = "\n".join([one_block] * repeat)

    return KERNEL_TEMPLATE.format(
        iters=iters,
        grid=grid,
        block=block,
        extra_args=", float* __restrict__ buf" if needs_buf else "",
        setup=spec["setup"],
        asm_blocks=asm_blocks,
        sink=spec["sink"],
        host_setup=(f"float* buf; cudaMalloc(&buf, {BUF_BYTES_PER_THREAD} * {grid} * {block});"
                    if needs_buf else ""),
        launch_extra=", buf" if needs_buf else "",
        host_teardown="cudaFree(buf);" if needs_buf else "",
    )


def _execute(insn, iters, repeat):
    src = build_program(insn, iters=iters, repeat=repeat)
    t0 = time.time()
    r = execute_code(src, nvcc_args=[], binary_args=[], enable_metrics=True, metrics_sleep_time=METRICS_WARMUP_S)
    ar = run_analyser(r.path, Path(WEIGHTS_PATH))
    print(f" {time.time()-t0:.1f}s total", flush=True)
    return r, ar


def run_one(insn, pilot_cache: dict):
    cache_key = insn
    if cache_key in pilot_cache:
        iters = pilot_cache[cache_key]
        print(f"  pilot cached -> {iters} iters", flush=True)
    else:
        print(f"  pilot ({PILOT_ITERS} iters)...", flush=True)
        pilot_r, _ = _execute(insn, iters=PILOT_ITERS, repeat=1)
        dur_ns = pilot_r.power_metric_result.kernel_duration_gpu_ns
        iters = max(PILOT_ITERS, int(PILOT_ITERS * TARGET_NS / max(dur_ns, 1.0)))
        print(f"  pilot done ({dur_ns*1e-9:.2f}s) -> {iters} iters", flush=True)
        pilot_cache[cache_key] = iters
        _save_pilot_cache(pilot_cache)

    print(f"  repeat=1 ({iters} iters)...", flush=True)
    r1, ar1 = _execute(insn, iters=iters, repeat=1)
    print(f"  repeat=2 ({iters} iters)...", flush=True)
    time.sleep(1)  # Short pause to ensure any lingering effects from the first run are minimized
    r2, ar2 = _execute(insn, iters=iters, repeat=2)

    r1_energy_per_op_j = r1.power_metric_result.total_energy_j / (iters * GRID * BLOCK)
    r2_energy_per_op_j = r2.power_metric_result.total_energy_j / (iters * GRID * BLOCK * 2) # 2x ops in repeat=2

    print(f"r1 (s): {r1.power_metric_result.kernel_duration_gpu_ns*1e-9:.2f} r1 (J/op): {r1_energy_per_op_j:.3e}", flush=True)
    print(f"r2 (s): {r2.power_metric_result.kernel_duration_gpu_ns*1e-9:.2f} r2 (J/op): {r2_energy_per_op_j:.3e}", flush=True)

    delta_energy_j = r2.power_metric_result.total_energy_j - r1.power_metric_result.total_energy_j
    err_delta_energy_per_op_j =  abs(r2_energy_per_op_j - r1_energy_per_op_j) / ((r2_energy_per_op_j + r1_energy_per_op_j) / 2) * 100


    print(f"delta energy (J): {delta_energy_j:.3e} delta energy/op (J): {delta_energy_j / (iters * GRID * BLOCK):.3e} err delta energy/op (%): {err_delta_energy_per_op_j:.2f}", flush=True)

    print(f"  raw run ({iters} iters)...", flush=True)

    return {
        "iters":               iters,

        "r1_energy_j":              r1.power_metric_result.total_energy_j,
        "r2_energy_j":              r2.power_metric_result.total_energy_j,

        "delta_energy_j":        delta_energy_j,
        "err_delta_energy_per_op_j": err_delta_energy_per_op_j,
        
        "r1_path": r1.path,
        "r2_path": r2.path,
    }


def main():
    pilot_cache = _load_pilot_cache()
    results = {}
    for insn in INSTRUCTION_TEMPLATES:
        print(f"\n=== {insn} ===")
        try:
            r = run_one(insn, pilot_cache)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        results[insn] = r

    print("\n\n" + "=" * 90)
    print(f"{'instruction':<16} {'delta J/op':>14} {'delta W':>8} {'delta s':>7} {'delta err%':>10}")
    print("=" * 90)
    for insn, r in results.items():
        print(f"{insn:<16} "
              f"{r['err_delta_energy_per_op_j']:>8.2f} "
        )

    out_path = Path(WEIGHTS_PATH)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["kernel_name", "instruction", "count", "total_instructions", "power_consumption_joules"])
        writer.writeheader()
        for insn, r in results.items():
            kernel_name = insn.replace(".", "_")
            count = r["iters"] * GRID * BLOCK # Total instructions is iters * threads, and we have 1 instruction per thread per iteration
            writer.writerow({
                "kernel_name":              kernel_name,
                "instruction":              insn,
                "count":                    count,
                "total_instructions":       count,
                "power_consumption_joules": r["delta_energy_j"],
            })
    print(f"\nwrote {len(results)} rows to {out_path}")

    return results


if __name__ == "__main__":
    results = main()
