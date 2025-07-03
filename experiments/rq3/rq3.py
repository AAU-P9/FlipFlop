#!/usr/bin/env python3
"""
RQ3 Comprehensive Experiment & Report Generator
===========================================================================
This integrated script:
1. Runs kernel tuning experiments with detailed data collection
2. Analyzes results in real-time
3. Generates a comprehensive report with actual metrics for LaTeX
4. Tracks carbon emissions using CodeCarbon
"""

import argparse
import os
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
import pycuda.driver as pcudadriver
import pycuda.autoinit
from collections import OrderedDict
from codecarbon import OfflineEmissionsTracker
import kernel_tuner as kt

from kernel_tuner.observers import BenchmarkObserver
from kernel_tuner.observers.nvml import NVMLObserver
from kernel_tuner.observers.ncu import NCUObserver

# Carbon conversion constants (kg CO2/kWh) - (https://ourworldindata.org/grapher/carbon-intensity-electricity)
CARBON_INTENSITY = 0.384

class WallTimeObserver(BenchmarkObserver):
    """Custom observer to measure total wall time per configuration"""
    def __init__(self):
        super().__init__()
        self.start_time = 0
        self.wall_time = 0
        self.name = "walltime_observer"
        
    def before_start(self):
        self.start_time = time.time()
        
    def after_finish(self):
        self.wall_time = time.time() - self.start_time
        
    def get_results(self):
        return {"wall_time": self.wall_time}

class ExperimentTracker:
    """Tracks experiment-wide metrics for comprehensive reporting"""
    def __init__(self):
        self.sequence_data = {}
        self.total_energy = 0
        self.total_time = 0
        self.carbon_tracker = None
        self.country_code = "USA"
        
    def start_experiment(self, country_code="USA"):
        self.country_code = country_code
        self.carbon_tracker = OfflineEmissionsTracker(
            country_iso_code=country_code,
            log_level="error",
            measure_power_secs=1,
            save_to_file=False
        )
        self.carbon_tracker.start()
        
    def add_sequence(self, seq_len, df):
        """Add sequence data to tracker"""
        if not df.empty:
            seq_energy = df["nvml_energy"].sum()
            seq_time = df["wall_time"].sum()
            pareto_configs = df["is_pareto"].sum()
            total_configs = len(df)
            
            self.sequence_data[seq_len] = {
                "total_configs": total_configs,
                "pareto_configs": pareto_configs,
                "crr": 1 - (pareto_configs / total_configs),
                "energy": seq_energy,
                "time": seq_time
            }
            self.total_energy += seq_energy
            self.total_time += seq_time
            
    def finalize_experiment(self):
        """Complete carbon tracking and final calculations"""
        if self.carbon_tracker:
            self.carbon_tracker.stop()
            self.total_carbon = self.carbon_tracker.final_emissions
        else:
            # Estimate carbon if tracking wasn't started
            energy_kwh = self.total_energy / 3.6e6
            self.total_carbon = energy_kwh * CARBON_INTENSITY
            
    def generate_report(self):
        """Generate comprehensive RQ3 report"""
        if not self.sequence_data:
            return "No data collected - report generation failed"
            
        report = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Header
        report.append(f"RQ3 Experimental Analysis Report")
        report.append(f"Generated: {timestamp}")
        report.append(f"Country Code: {self.country_code} (Carbon Intensity: {CARBON_INTENSITY} kg CO2/kWh)")
        report.append("=" * 80)
        report.append("")
        
        # List all sequence lengths
        seq_lens_list = sorted(self.sequence_data.keys())
        report.append(f"Sequence lengths: {', '.join(str(s) for s in seq_lens_list)}")
        report.append("")
        
        # Configuration Reduction Analysis
        report.append("1. Configuration Reduction Analysis")
        report.append("-" * 80)
        report.append(f"Sequence lengths analyzed: {len(self.sequence_data)}")
        
        # Detailed CRR table
        report.append("Per-sequence length results:")
        report.append("Seq Len  Total Configs  Pareto Configs  CRR      Time (s)   Time (us)   Energy (J)  Time/Run (s)")
        report.append("------  -------------  --------------  -------  ---------  ----------  -----------  ------------")
        for seq in seq_lens_list:
            data = self.sequence_data[seq]
            time_us = data['time'] * 1e6
            # Calculate average time per run (kernel execution time)
            time_per_run = data['time'] / (data['total_configs'] * 5)  # 5 iterations per config
            report.append(f"{seq:6}  {data['total_configs']:13}  {data['pareto_configs']:15}  {data['crr']:.3f}  {data['time']:9.1f}  {time_us:10.1f}  {data['energy']:11.1f}  {time_per_run:12.2f}")
        
        # Summary statistics
        avg_crr = np.mean([d["crr"] for d in self.sequence_data.values()])
        avg_pareto = np.mean([d["pareto_configs"] for d in self.sequence_data.values()])
        report.append("")
        report.append(f"Average Pareto-optimal configs per seq: {avg_pareto:.1f}")
        report.append(f"Average CRR: {avg_crr:.3f}")
        
        # Calculate profiling time per sequence length (as in manuscript)
        avg_time_per_run = np.mean([data['time'] / (data['total_configs'] * 5) for data in self.sequence_data.values()])
        avg_configs_per_seq = np.mean([data['total_configs'] for data in self.sequence_data.values()])
        time_per_config = avg_time_per_run * 5  # 5 iterations per config
        time_per_seq = time_per_config * avg_configs_per_seq
        total_profiling_time = time_per_seq * len(self.sequence_data)
        
        report.append("")
        report.append("Profiling Time Analysis (per manuscript):")
        report.append(f"  Average time per run: {avg_time_per_run:.2f} seconds")
        report.append(f"  Time per configuration (5 runs): {time_per_config:.2f} seconds")
        report.append(f"  Time per sequence length ({avg_configs_per_seq:.0f} configs): {time_per_seq/60:.1f} minutes")
        report.append(f"  Total profiling time ({len(self.sequence_data)} seq lengths): {total_profiling_time/60:.1f} minutes")
        report.append("")
        
        # Resource Savings
        report.append("2. Resource Savings")
        report.append("-" * 80)
        
        # Calculate brute-force vs FlipFlop savings
        total_ff_time = sum(min(d["time"], d["time"] * d["pareto_configs"] / d["total_configs"]) 
                          for d in self.sequence_data.values())
        total_ff_energy = sum(min(d["energy"], d["energy"] * d["pareto_configs"] / d["total_configs"]) 
                           for d in self.sequence_data.values())
        
        time_savings = self.total_time - total_ff_time
        energy_savings = self.total_energy - total_ff_energy
        
        # Carbon calculations
        ff_carbon = total_ff_energy / 3.6e6 * CARBON_INTENSITY
        carbon_savings = self.total_carbon - ff_carbon
        
        report.append(f"Total experiment time: {self.total_time/60:.1f} minutes")
        report.append(f"Total experiment energy: {self.total_energy:.1f} Joules")
        report.append(f"Total carbon emissions: {self.total_carbon*1000:.1f} g CO₂e")
        report.append("")
        report.append(f"Estimated FlipFlop time: {total_ff_time/60:.1f} minutes")
        report.append(f"Estimated FlipFlop energy: {total_ff_energy:.1f} Joules")
        report.append(f"Estimated FlipFlop carbon: {ff_carbon*1000:.1f} g CO₂e")
        report.append("")
        report.append(f"Time savings: {time_savings/60:.1f} minutes")
        report.append(f"Energy savings: {energy_savings:.1f} Joules")
        report.append(f"Carbon savings: {carbon_savings*1000:.1f} g CO₂e")
        report.append("")
        
        # Developer Impact Projection
        report.append("3. Developer Impact Projection")
        report.append("-" * 80)
        
        # For 5 sequence lengths
        avg_time_per_seq = self.total_time / len(self.sequence_data)
        avg_energy_per_seq = self.total_energy / len(self.sequence_data)
        
        time_5seq_bf = avg_time_per_seq * 5
        time_5seq_ff = avg_time_per_seq * 5 * (avg_pareto / np.mean([d["total_configs"] for d in self.sequence_data.values()]))
        
        # For 100-kernel model
        time_100k_bf = avg_time_per_seq * 100
        time_100k_ff = time_100k_bf * (avg_pareto / np.mean([d["total_configs"] for d in self.sequence_data.values()]))
        
        energy_100k_bf = avg_energy_per_seq * 100
        energy_100k_ff = energy_100k_bf * (avg_pareto / np.mean([d["total_configs"] for d in self.sequence_data.values()]))
        energy_100k_savings = energy_100k_bf - energy_100k_ff
        carbon_100k_savings = energy_100k_savings / 3.6e6 * CARBON_INTENSITY
        
        report.append(f"Iterative tuning (5 seq lengths):")
        report.append(f"  Brute-force: {time_5seq_bf/60:.1f} minutes")
        report.append(f"  FlipFlop: {time_5seq_ff/60:.1f} minutes")
        report.append(f"  Speedup: {time_5seq_bf/time_5seq_ff:.1f}x")
        report.append("")
        
        report.append(f"100-kernel model optimization:")
        report.append(f"  Brute-force: {time_100k_bf/60:.1f} minutes")
        report.append(f"  FlipFlop: {time_100k_ff/60:.1f} minutes")
        report.append(f"  Energy savings: {energy_100k_savings:.1f} Joules")
        report.append(f"  Carbon savings: {carbon_100k_savings*1000:.1f} g CO₂e")
        report.append("")
        
        # Key Metrics for LaTeX
        report.append("4. Key Metrics for LaTeX")
        report.append("-" * 80)
        report.append(f"CRR_avg = {avg_crr:.3f}  # Average Configuration Reduction Ratio")
        report.append(f"Time_speedup = {self.total_time/total_ff_time:.1f}  # Overall speedup factor")
        report.append(f"Energy_savings_total = {energy_savings:.1f}  # Total energy savings in Joules")
        report.append(f"Carbon_savings_total = {carbon_savings*1000:.6f}  # Total carbon savings in grams CO₂e")
        report.append(f"Pareto_avg = {avg_pareto:.1f}  # Average Pareto-optimal configurations")
        
        return "\n".join(report)

def run_tuning(kernel_string, batch_size, seq_len, nhead, dim_per_head,
               csv_out, iterations, strategy, label, tracker):
    """Enhanced tuning routine for RQ3 data collection"""
    dim_feature = nhead * dim_per_head
    scale = np.float32(1.0 / np.sqrt(dim_per_head))

    # Prepare input data
    q = np.random.randn(batch_size, dim_feature).astype(np.float32).ravel()
    k = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32).ravel()
    v = np.random.randn(batch_size, seq_len, dim_feature).astype(np.float32).ravel()
    out = np.zeros_like(q)

    arguments = [
        q, k, v,
        np.int32(batch_size),
        np.int32(seq_len),
        np.int32(dim_feature),
        np.int32(dim_feature),
        np.int32(nhead),
        scale,
        np.int32(64),
        out,
    ]

    # Tuning parameters (block configurations only for RQ3)
    tune_params = {
        "block_size_x": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        "block_size_y": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    }

    # tune_params = {
    #     "block_size_x": [1, 2],
    #     "block_size_y": [1, 2],
    # }

    # Restrict to valid block configurations
    # restrictions = [
    #     "(block_size_x * block_size_y) >= 32",
    #     "(block_size_x * block_size_y) <= 1024",
    #     "(block_size_x * block_size_y) % 32 == 0"
    # ]

    # Observers for comprehensive data collection
    nsight_metrics = [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.max.pct_of_peak_sustained_active",
        "sm__cycles_active.avg.pct_of_peak_sustained_active",
    ]
    observers = [
        WallTimeObserver(),
        NVMLObserver(["nvml_energy", "nvml_power", "temperature"]),
        NCUObserver(metrics=nsight_metrics)
    ]

    # Calculate derived metrics
    total_ops = 2.0 * dim_feature * seq_len * batch_size

    # Enhanced metrics for RQ3 analysis
    metrics = OrderedDict()
    metrics["time_per_run"] = lambda p: p["time"]  # Kernel execution time per run
    metrics["energy_per_run"] = lambda p: p.get("nvml_energy", 0.0) / iterations
    metrics["wall_time"] = lambda p: p.get("wall_time", 0.0)
    metrics["Joules/token"] = lambda p: (p.get("nvml_energy", 0.0) / (batch_size * seq_len * iterations))
    metrics["FLOPS/Watt"] = lambda p: (total_ops / p.get("nvml_energy", 0.0)) * iterations if p.get("nvml_energy", 0.0) > 0 else 0.0
    metrics["SM_Active_Avg"] = lambda p: p.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)

    compiler_options = [
        "--std=c++14",
        "-I/usr/local/cuda/include",
        "-I/usr/local/cuda/include/cub",
    ]

    problem_size = (batch_size * nhead,)
    shared_mem_size = (dim_per_head + seq_len) * 4

    # Kernel Tuner call with enhanced data collection
    results, env = kt.tune_kernel(
        kernel_name="mha",
        kernel_source=kernel_string,
        problem_size=problem_size,
        arguments=arguments,
        tune_params=tune_params,
        observers=observers,
        metrics=metrics,
        strategy=strategy,
        # restrictions=restrictions,
        compiler_options=compiler_options,
        smem_args={"size": shared_mem_size},
        iterations=iterations,
        verbose=True,
        cache=None,  # Disable cache for accurate measurements
        objective="Joules/token",
        objective_higher_is_better=False
    )

    # Post-process results for Pareto-optimal configurations
    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Keep any point with a lower cost in both dimensions
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
                is_efficient[i] = True
        return is_efficient

    if results:
        # Extract performance (time) and efficiency (energy) metrics
        times = np.array([r['time_per_run'] for r in results])
        energies = np.array([r['energy_per_run'] for r in results])
        costs = np.column_stack([times, energies])
        
        # Find Pareto-optimal configurations
        pareto_mask = is_pareto_efficient(costs)
        for i, r in enumerate(results):
            r["is_pareto"] = bool(pareto_mask[i])
    else:
        print(f"[WARNING] No results collected for seq_len={seq_len}")
        return pd.DataFrame()

    # Add metadata
    timestamp = datetime.now().isoformat()
    for r in results:
        r.update({
            "timestamp": timestamp,
            "run_label": label,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "nhead": nhead,
            "dim_per_head": dim_per_head,
            "strategy": strategy,
            "iterations": iterations
        })

    # Return results DataFrame (don't save here - will be saved in main)
    df = pd.DataFrame(results)
    
    print(f"[RQ3] Collected {len(results)} configs for seq_len={seq_len} | "
          f"Pareto-optimal: {sum(pareto_mask)}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="RQ3: Comprehensive Experiment & Report Generator")
    parser.add_argument("--kernel_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_lens", type=str, default="128,256,512,786,1024,2048,4096,8192")
    parser.add_argument("--nhead", type=int, default=16)
    parser.add_argument("--dim_per_head", type=int, default=256)
    parser.add_argument("--csv_out", type=str, default="rq3_data/mha_results.csv")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--strategy", type=str, default="brute_force")
    parser.add_argument("--label", type=str, default="rq3_experiment")
    parser.add_argument("--report_out", type=str, default="rq3_report.txt")
    parser.add_argument("--country_code", type=str, default="USA", 
                        help="Country code for carbon emission calculations")
    args = parser.parse_args()

    # Initialize experiment tracker
    tracker = ExperimentTracker()
    tracker.start_experiment(args.country_code)
    
    # Parse sequence lengths
    seq_lens = [int(s) for s in args.seq_lens.split(",")] if args.seq_lens else [128]

    # Load kernel source
    with open(args.kernel_file, "r") as f:
        kernel_src = f.read()

    # Collect all results
    all_results = []
    
    # Run tuning for each sequence length
    for sl in seq_lens:
        print(f"\n{'='*60}")
        print(f"Starting RQ3 experiment for seq_len={sl}")
        print(f"{'='*60}")
        
        df = run_tuning(
            kernel_string=kernel_src,
            batch_size=args.batch_size,
            seq_len=sl,
            nhead=args.nhead,
            dim_per_head=args.dim_per_head,
            csv_out=args.csv_out,
            iterations=args.iterations,
            strategy=args.strategy,
            label=args.label,
            tracker=tracker
        )
        
        if not df.empty:
            all_results.append(df)
            df_seq = df[df["seq_len"] == sl]
            total_configs = len(df_seq)
            tracker.add_sequence(sl, df_seq)
    
    # Save all results to CSV (always create new file)
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
        combined_df.to_csv(args.csv_out, index=False)
        print(f"[RQ3] Saved {len(combined_df)} total configurations to {args.csv_out}")
    
    # Finalize experiment and generate report
    tracker.finalize_experiment()
    report = tracker.generate_report()
    
    # Save report
    with open(args.report_out, "w") as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"RQ3 EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Report generated at: {args.report_out}")
    print(f"Data saved to: {args.csv_out}")
    print(f"Total experiment time: {tracker.total_time/60:.1f} minutes")
    print(f"Total energy consumed: {tracker.total_energy:.1f} Joules")
    print(f"Carbon emissions: {tracker.total_carbon*1000:.1f} g CO₂e")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()