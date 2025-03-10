#!/usr/bin/env python
"""
Energy-aware GPU kernel tuning using Kernel Tuner
"""

import os
import argparse
import numpy as np
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from kernel_tuner import tune_kernel
from kernel_tuner.observers.nvml import NVMLObserver, get_nvml_gr_clocks

class EnergyTuner:
    def __init__(self, kernel_path: str, device: int = 0):
        self.device = device
        self.kernel_path = Path(kernel_path)
        self.kernel_string = self._read_kernel()
        
        # Setup NVML observer
        self.observer = NVMLObserver(
            ["core_freq", "nvml_power"], 
            device=device
        )
        
        # Energy metrics
        self.metrics = OrderedDict([
            ("time", lambda p: p["time"]),
            ("power", lambda p: p["nvml_power"]), 
            ("energy", lambda p: p["nvml_power"] * p["time"]/1000),
            ("efficiency", lambda p: self.compute_gflops(p)/(p["nvml_power"] * p["time"]/1000))
        ])

    def compute_gflops(self, p: Dict) -> float:
        N = p.get("size", 1024)
        ops = 2.0 * (N**3)
        return (ops / 1e9) / (p["time"] / 1000.0)

    def _read_kernel(self) -> str:
        if not self.kernel_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {self.kernel_path}")
            
        with open(self.kernel_path) as f:
            return f.read()

    def prepare_data(self, problem_size: Tuple[int, ...]) -> List[np.ndarray]:
        N = problem_size[0]
        A = np.random.randn(N*N).astype(np.float32)
        B = np.random.randn(N*N).astype(np.float32)
        C = np.zeros(N*N, dtype=np.float32)
        return [C, A, B, np.int32(N)]

    def get_tune_params(self) -> Dict:
        return OrderedDict([
            ("block_size_x", [16, 32, 64, 128, 256]),
            ("block_size_y", [1, 2, 4, 8, 16]),
            ("use_shared_mem", [0, 1]),
            ("unroll_factor", [1, 2, 4, 8])
        ])

    def get_restrictions(self):
        return ["block_size_x * block_size_y <= 1024",  # Max threads per block
                "block_size_x >= 32",                    # Min warp size
                "block_size_y >= 1"]                     # Min y dimension

    def get_available_frequencies(self) -> List[int]:
        clocks = get_nvml_gr_clocks(self.device, quiet=True)
        return clocks['nvml_gr_clock']

    def tune(self, problem_size: Tuple[int, ...], iterations: int = 10) -> List[Dict]:
        available_freqs = self.get_available_frequencies()
        
        test_freqs = np.linspace(
            min(available_freqs), 
            max(available_freqs), 
            5, 
            dtype=int
        ).tolist()
        
        args = self.prepare_data(problem_size)
        tune_params = self.get_tune_params()
        
        results = []
        for freq in test_freqs:
            print(f"\nTuning at {freq} MHz...")
            
            # Update NVML parameters for this frequency
            tune_params.update({"nvml_gr_clock": [freq]})
            
            try:
                freq_results = tune_kernel(
                    kernel_name="kernel_func",
                    kernel_source=self.kernel_string,
                    problem_size=problem_size,
                    arguments=args,
                    tune_params=tune_params,
                    grid_div_x=["block_size_x"],
                    grid_div_y=["block_size_y"],
                    metrics=self.metrics,
                    observers=[self.observer],
                    iterations=iterations,
                    verbose=True,
                    restrictions=self.get_restrictions(),
                    quiet=False,
                    lang="CUDA",
                    compiler_options=["-arch=sm_86"],
                    cache="energy_tuning_cache.json"

                )
                
                for result in freq_results:
                    result["size"] = problem_size[0]
                
                results.extend(freq_results)
                
            except Exception as e:
                print(f"Error tuning at frequency {freq} MHz: {str(e)}")
                continue
        
        return results

    def analyze_results(self, results: List[Dict]) -> None:
        if not results:
            print("No results to analyze!")
            return
            
        # Get Pareto-optimal configurations
        pareto_front = []
        for config in sorted(results, key=lambda x: (x["time"], x["energy"])):
            if not any(c["time"] <= config["time"] and c["energy"] <= config["energy"] 
                      for c in pareto_front):
                pareto_front.append(config)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        times = [r["time"] for r in results]
        energies = [r["energy"] for r in results]
        plt.scatter(times, energies, alpha=0.5, label="All configs")
        
        pareto_times = [r["time"] for r in pareto_front]
        pareto_energies = [r["energy"] for r in pareto_front]
        plt.scatter(pareto_times, pareto_energies, color='red', label="Pareto front")
        
        plt.xlabel("Execution Time (ms)")
        plt.ylabel("Energy Consumption (J)")
        plt.title("Energy-Performance Trade-off")
        plt.legend()
        plt.grid(True)
        plt.savefig("energy_tuning_results.pdf")
        
        # Print best configurations
        print("\nBest configurations:")
        print("\nFastest:")
        best_time = min(results, key=lambda x: x["time"])
        print(f"Time: {best_time['time']:.2f} ms")
        print(f"Energy: {best_time['energy']:.2f} J")
        print(f"Config: {best_time}")
        
        print("\nMost energy efficient:")
        best_energy = min(results, key=lambda x: x["energy"])
        print(f"Time: {best_energy['time']:.2f} ms")
        print(f"Energy: {best_energy['energy']:.2f} J")
        print(f"Config: {best_energy}")
        
        print("\nBest efficiency (GFLOPS/J):")
        best_eff = max(results, key=lambda x: x["efficiency"])
        print(f"Efficiency: {best_eff['efficiency']:.2f} GFLOPS/J")
        print(f"Config: {best_eff}")

def main():
    parser = argparse.ArgumentParser(description="Energy-aware GPU kernel tuning")
    parser.add_argument("kernel_file", help="Path to CUDA kernel file")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--size", type=int, default=1024, help="Problem size")
    parser.add_argument("--iterations", type=int, default=10, 
                       help="Iterations per configuration")
    args = parser.parse_args()

    tuner = EnergyTuner(args.kernel_file, args.device)
    results = tuner.tune((args.size, args.size), args.iterations)
    tuner.analyze_results(results)

if __name__ == "__main__":
    main()