import subprocess
import os
import signal
import time
import sys
from pathlib import Path
import shutil
import re
from dataclasses import dataclass
from cubindings_power import extract_power_metrics, PowerMetricsResult
from cubindings_types import ExportJSONResponse
import json

def extract_exports_from_path(path: Path) -> ExportJSONResponse:
    output_json = path / "output.json"
    if output_json.exists():
        with output_json.open("r") as f:
            exports = json.load(f)
    return exports

def _resolve_pmd2_cli_path() -> str | None:
    env_path = os.environ.get("PMD2_CLI_PATH")
    if env_path:
        return env_path

    workspace_cli = Path(__file__).parents[1] / "libEPMD" / "cli" / "pmd2-cli"
    if workspace_cli.exists() and os.access(workspace_cli, os.X_OK):
        return str(workspace_cli)

    return shutil.which("pmd2-cli")

@dataclass
class ExecutionResult:
    output: str
    error: str
    exports: ExportJSONResponse
    returncode: int
    path: Path
    power_metric_result: PowerMetricsResult = None

def _terminate_process_group(
    process: subprocess.Popen | None,
    terminate_signal: int = signal.SIGTERM,
    wait_timeout_s: float = 2.0,
) -> None:
    if process is None:
        return

    if process.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(process.pid), terminate_signal)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=wait_timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()
    
def execute_code(
    program_str: str,
    path: Path = None,
    nvcc_args: list[str] = [], binary_args: list[str] = [],
    enable_metrics: bool = False,
    metrics_sleep_time: int = 5,
) -> ExecutionResult:
    # Fallback temporary directory if the specified path cannot be created
    if path is None:
        path = Path(f"/tmp/{time.time()}")
        path.mkdir(parents=True, exist_ok=True)

    # Prepend the cupti_timing include to the program string    
    program_str = "#include \"cupti_timing.h\"\n" + program_str

    # Write the program string to a temporary file
    program_file = path / "program.cu"

    # Append the metrics code to the program string if metrics are enabled
    if enable_metrics:
        pattern = r'(int\s+main\s*\([^)]*\)\s*\{)(.*?)(return\s+0\s*;\s*\})'
        program_str = re.sub(
            pattern,
            r'\1\nMETRICS_KERNEL_START\n\2\nMETRICS_KERNEL_END\n\3',
            program_str,
            flags=re.DOTALL
        )

    with program_file.open("w") as f:
        f.flush() # Ensure the file is created and flushed before writing to it
        f.write(program_str)
    
    return execute_program(
        path=path,
        nvcc_args=nvcc_args,
        binary_args=binary_args,
        enable_metrics=enable_metrics,
        metrics_sleep_time=metrics_sleep_time,
    )

def execute_program(
    path: Path,
    nvcc_args: list[str] = [],
    binary_args: list[str] = [],
    enable_compilation: bool = True,
    enable_metrics: bool = True,
    enable_temp: bool =True,
    metrics_sleep_time: int = 5,
    program_name: str = "program.cu",
    debug: bool = False,
) -> ExecutionResult:   
    pipe = sys.stdout if debug else subprocess.PIPE

    if enable_temp:
        tmp_dir = Path(f"/tmp/{time.time()}")
        shutil.copytree(path, tmp_dir)
        path = tmp_dir

    program_file = path / program_name

    # Initialize power_metric_result to None by default
    power_metric_result = None

    # Absolute path to the include directory for cubindings.h
    nvcc_cmd = ["nvcc", "-Xptxas", "-g", "-G", "-O0", "-arch=sm_89", "-lcupti"]
    include_path = Path(__file__).parents[1] / "include"
    nvcc_cmd.append(f"-I{str(include_path)}")
    nvcc_cmd.extend(list(nvcc_args))
    nvcc_cmd.append(str(program_file))
    nvcc_cmd.append("--keep")

    if enable_compilation:
        nvcc_process = subprocess.run(
            nvcc_cmd,
            cwd=path, # Set the current working directory to the temporary directory
            stdout=pipe,
            stderr=pipe,
            text=True,
            check=False,
        )
        if nvcc_process.returncode != 0:
            raise RuntimeError(
                "NVCC compilation failed:\n"
                f"{' '.join(nvcc_cmd)}\n"
                f"{(nvcc_process.stderr or '').strip()}"
            )
        
    # Start monitoring processes if metrics are enabled
    monitor_process = None
    pmd2_process = None
    if enable_metrics:
        # Start nvidia-smi monitoring on a separate process
        monitor_log = path / "nvidia-smi.csv"
        monitor_process = None
        
        # Start pmd2-cli monitoring on a separate process
        pmd2_log = path / "pmd2.csv"
        pmd2_process = None
        if shutil.which("nvidia-smi") is not None:
            try:
                with open(monitor_log, "w") as monitor_file:
                    monitor_file.flush() # Ensure the file is created and flushed before nvidia-smi tries to write to it
                    monitor_process = subprocess.Popen(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,power.draw,power.limit,temperature.gpu,fan.speed",
                            "--format=csv",
                            "-lms",
                            "5",
                        ],
                        stdout=monitor_file,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
            except Exception as e:
                print(f"Warning: Failed to start nvidia-smi monitoring: {e}", file=sys.stderr)
                monitor_process = None
        else:
            print("Warning: nvidia-smi not found in PATH, skipping nvidia-smi monitoring", file=sys.stderr)
        
        pmd2_cli = _resolve_pmd2_cli_path()
        if pmd2_cli is not None:
            subprocess.run(
                ["pkill", "-x", "-u", str(os.getuid()), "pmd2-cli"],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            try:
                pmd2_cmd = [
                    pmd2_cli,
                    "-p",
                    "/dev/ttyACM0",
                    "-c",
                    "-i",
                    "100",
                    "monitor",
                ]
                # PMD2 may keep stdout block-buffered when redirected to a file.
                # Line buffering avoids empty CSV files on short runs.
                if shutil.which("stdbuf") is not None:
                    pmd2_cmd = ["stdbuf", "-oL", *pmd2_cmd]
                with open(pmd2_log, "w") as pmd2_file:
                    pmd2_file.flush() # Ensure the file is created and flushed before pmd2-cli tries to write to it
                    pmd2_process = subprocess.Popen(
                        pmd2_cmd,
                        stdout=pmd2_file,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,
                    )
            except Exception as e:
                print(f"Warning: Failed to start pmd2-cli monitoring: {e}", file=sys.stderr)
                pmd2_process = None

        else:
            print("Warning: pmd2-cli not found in PATH, skipping PMD2 monitoring", file=sys.stderr)

    bin_cmd = [str(path / "a.out")]
    bin_cmd.extend(binary_args)

    execution_process = None
    exports = {}

    try:
        if enable_metrics:
            # Sleep for the specified duration to ensure subprocesses have time to collect data
            time.sleep(metrics_sleep_time)

            if pmd2_process is not None and pmd2_process.poll() is not None:
                stderr_out = pmd2_process.stderr.read()
                print(f"Warning: pmd2-cli exited early (code {pmd2_process.returncode})", file=sys.stderr)
                if stderr_out:
                    print(f"pmd2-cli stderr: {stderr_out}", file=sys.stderr)

        # Run the compiled program and capture its output
        execution_process = subprocess.run(
            bin_cmd,
            cwd=path, # Set the current working directory to the temporary directory
            stdout=pipe,
            stderr=pipe,
            text=True,
            check=False,
        )

        if execution_process.returncode != 0:
            print(
                "[WARNING]: Program execution failed:\n"
                f"{' '.join(bin_cmd)}\n"
                f"{(execution_process.stderr or '').strip()}",
                file=sys.stderr,
            )

        # Read the output.json file to get the exported variables
        exports = extract_exports_from_path(path)
        
    finally:
        if enable_metrics:
            _terminate_process_group(monitor_process)
            _terminate_process_group(pmd2_process)
            if pmd2_process is not None and pmd2_log.stat().st_size == 0:
                stderr_out = pmd2_process.stderr.read()
                print(f"Warning: pmd2-cli wrote nothing (exit code {pmd2_process.returncode})", file=sys.stderr)
                if stderr_out:
                    print(f"pmd2-cli stderr: {stderr_out}", file=sys.stderr)
            power_metric_result = extract_power_metrics(path, exports)

    # Return the execution result
    return ExecutionResult(
        output=(execution_process.stdout if execution_process is not None else ""),
        error=(execution_process.stderr if execution_process is not None else ""),
        exports=exports,
        returncode=(execution_process.returncode if execution_process is not None else -1),
        power_metric_result=power_metric_result,
        path=path
    )