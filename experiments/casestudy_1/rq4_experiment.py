#!/usr/bin/env python3
"""
Simplified evaluation script for RQ4 energy comparison
"""

import os
import time
import random
import csv
import subprocess
import numpy as np
import yaml
from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex, 
                    nvmlDeviceGetPowerUsage, nvmlShutdown)

# Configuration
BASELINE_BIN = "/home/srajput/flipflop/llama3.cuda/runcuda"
TUNED_BIN = "/home/srajput/flipflop/llama3.cuda/runcuda_tuned"
NUM_EXAMPLES = 44
MAX_GEN_TOKENS = 64
SEED = 42
random.seed(SEED)
POWER_SAMPLING_INTERVAL = 0.01

class PowerMonitor:
    """Simplified power monitor without limit control"""
    def __init__(self):
        self.power_readings = []
        
    def __enter__(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        return self
    
    def __exit__(self, *args):
        nvmlShutdown()
        
    def measure(self):
        """Run measurement during inference"""
        self.power_readings = []
        start_time = time.time()
        while time.time() - start_time < 10:  # Timeout safety
            try:
                self.power_readings.append(
                    nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                )
                time.sleep(POWER_SAMPLING_INTERVAL)
            except:
                break
        return np.mean(self.power_readings) if self.power_readings else 0.0

def run_inference(model_bin, prompt):
    """Run inference with power measurement"""
    with PowerMonitor() as pmon:
        cmd = [model_bin, prompt]
        try:
            start_time = time.perf_counter()
            
            # Run with bytes output and safe decoding
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True
            )
            output = result.stdout.decode('utf-8', errors='replace').strip()
            
            elapsed = time.perf_counter() - start_time
            avg_power = pmon.measure()
            return output, elapsed, avg_power
            
        except subprocess.CalledProcessError as e:
            print(f"Error running {model_bin}: {e}")
            return "", 0.0, 0.0

def main(model_bin, power_limit, block_size):
    with open("Evaluation_prompts.yaml") as f:
        eval_prompts = yaml.safe_load(f)
    
    indices = random.sample(range(len(eval_prompts)), NUM_EXAMPLES)

    config_name = os.path.basename(model_bin)
    
    csv_file = f"tinystories_{config_name}_{power_limit}W_results.csv"
    with open(csv_file, "w") as outf:
        writer = csv.DictWriter(outf, fieldnames = [
            "example_id", "prompt_len", "gen_text", 
            "time_s", "power_w", "gen_tokens",
            "block_size", "power_limit", "config_name"
        ])
        if outf.tell() == 0:
            writer.writeheader()
            
        for i, idx in enumerate(indices):
            prompt = eval_prompts[idx].strip()
            
            gen_text, elapsed, avg_power = run_inference(model_bin, prompt)
            
            writer.writerow({
                "example_id": idx,
                "prompt_len": len(prompt.split()),
                "gen_text": gen_text,
                "time_s": round(elapsed, 3),
                "power_w": round(avg_power, 1),
                "gen_tokens": len(gen_text.split()),
                "block_size": block_size,
                "config_name": config_name,
                "power_limit": power_limit
            })
            print(f"Processed {i+1}/{NUM_EXAMPLES}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="Path to the binary to run")
    parser.add_argument("--power_limit", type=int, required=True, help="Power limit in watts")
    parser.add_argument("--block_size", type=int, required=True, help="Kernel block size")
    args = parser.parse_args()


    
    main(args.binary, args.power_limit, args.block_size)