#!/usr/bin/env python3
"""
analysis_rq2.py

This script analyzes the experimental CSV data for RQ2:
Real-time Adaptation of Power Limits and Block Dimensions.

Objective:
  To evaluate whether jointly tuning the GPU power limit and 
  block dimensions can reduce energy consumption in LLM inference.
  
The script performs two main functions:
  1. It creates subplots (one per sequence length) of the best configuration
     (lowest Joules/token and its throughput) versus the applied power limit.
  2. It computes and prints aggregated statistics (mean and std for Joules/token
     and throughput) per power limit and sequence length, and writes these results 
     to a text report that also includes, for each sequence length, the best config 
     and the energy saving (both absolute and as a percentage) relative to the default
     power cap of 250 W.

Usage:
  python analysis_rq2.py <csv_file>
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_throughput(df):
    """
    Compute throughput (tokens/sec) from available columns.
    Assumes that 'time' is in seconds and that both 'batch_size' and 'seq_len'
    are present.
    """
    required_cols = ["time", "batch_size", "seq_len"]
    if all(col in df.columns for col in required_cols):
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce")
        df["seq_len"] = pd.to_numeric(df["seq_len"], errors="coerce")
        valid_mask = df["time"] > 0
        df.loc[valid_mask, "throughput"] = (df.loc[valid_mask, "batch_size"] *
                                            df.loc[valid_mask, "seq_len"]) / df.loc[valid_mask, "time"]
        df.loc[~valid_mask, "throughput"] = np.nan
    else:
        df["throughput"] = np.nan
    return df

def find_best_configs(df):
    """
    For each combination of (nvml_pwr_limit, seq_len), find the row
    with the minimal Joules/token.
    """
    grouping = ["nvml_pwr_limit", "seq_len"]
    valid_df = df.dropna(subset=["Joules/token"])
    valid_df = valid_df[valid_df["Joules/token"] > 0]
    if valid_df.empty:
        return pd.DataFrame()
    idx = valid_df.groupby(grouping)["Joules/token"].idxmin()
    best_configs = valid_df.loc[idx].copy()
    best_configs.sort_values(by=["seq_len", "nvml_pwr_limit"], inplace=True)
    return best_configs

def find_best_config_by_seq(df):
    """
    For each sequence length, select the overall best configuration (lowest Joules/token).
    """
    valid_df = df.dropna(subset=["Joules/token"])
    valid_df = valid_df[valid_df["Joules/token"] > 0]
    if valid_df.empty:
        return pd.DataFrame()
    idx = valid_df.groupby("seq_len")["Joules/token"].idxmin()
    best_by_seq = valid_df.loc[idx].copy()
    best_by_seq.sort_values("seq_len", inplace=True)
    return best_by_seq

def plot_best_config_vs_powerlimit_subplots(best_configs, out_dir="rq2_plots"):
    """
    Creates subplots (one per sequence length) showing the best configuration 
    vs. power limit.
    
    For each sequence length, a subplot is generated with:
      - Left y-axis: Joules/token (solid line with marker "o")
      - Right y-axis: Throughput (dashed line with marker "s")
    """
    if best_configs.empty:
        print("[WARN] No best configurations found. Skipping plot.")
        return

    seq_lens = sorted(best_configs["seq_len"].unique())
    n = len(seq_lens)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4*nrows), squeeze=False)
    fig.suptitle("Best Configuration vs. Power Limit by Sequence Length", fontsize=16, y=0.98)

    palette = sns.color_palette("Dark2", n_colors=n)

    for i, seq in enumerate(seq_lens):
        ax = axes[i // ncols][i % ncols]
        subset = best_configs[best_configs["seq_len"] == seq].sort_values("nvml_pwr_limit")
        color = palette[i]
        ax.plot(subset["nvml_pwr_limit"], subset["Joules/token"],
                marker="o", color=color, label=f"Joules/token (seq_len={int(seq)})")
        ax.set_xlabel("Power Limit (W)")
        ax.set_ylabel("Joules/token", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.set_title(f"Sequence Length: {int(seq)}")
        
        ax2 = ax.twinx()
        ax2.plot(subset["nvml_pwr_limit"], subset["throughput"],
                 marker="s", linestyle="--", color=color, label=f"Throughput (seq_len={int(seq)})")
        ax2.set_ylabel("Throughput (tokens/sec)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
        
    # Remove extra axes if any
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j // ncols][j % ncols])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(out_dir, "best_config_vs_powerlimit_by_seq_len_subplots.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved subplot figure to {outpath}")

def generate_text_report(df, best_configs, best_by_seq, out_dir="rq2_plots"):
    """
    Generate a text report including:
      - Aggregated statistics by (nvml_pwr_limit, seq_len).
      - Best configurations (lowest Joules/token) per (nvml_pwr_limit, seq_len).
      - For each sequence length, the best overall configuration and
        the energy saving (absolute and % difference) compared to the default 250W.
    The report is saved to 'analysis_report_rq2.txt' in out_dir.
    """
    grouping = ["nvml_pwr_limit", "seq_len"]
    agg_stats = df.groupby(grouping).agg({
        "Joules/token": ["mean", "std"],
        "throughput":   ["mean", "std"]
    })
    agg_stats.columns = ["_".join(col) for col in agg_stats.columns]
    agg_stats = agg_stats.reset_index()

    report_file = os.path.join(out_dir, "analysis_report_rq2.txt")
    with open(report_file, "w") as f:
        f.write("RQ2 Analysis Report\n")
        f.write("===================\n\n")
        f.write("Aggregated Statistics by (Power Limit, Seq. Len):\n\n")
        f.write(agg_stats.to_string(index=False))
        f.write("\n\nBest Configurations (Lowest Joules/token) by (Power Limit, Seq. Len):\n\n")
        best_cols = ["nvml_pwr_limit", "seq_len", "block_size_x", "block_size_y", "Joules/token", "throughput"]
        if not best_configs.empty:
            f.write(best_configs[best_cols].to_string(index=False))
        else:
            f.write("[No best configurations found]\n")
        f.write("\n\nBest Configurations per Sequence Length and Energy Savings vs. 250W:\n\n")
        # For each seq_len, compare the best config against the baseline at 250W.
        for seq in sorted(df["seq_len"].unique()):
            f.write(f"Sequence Length: {int(seq)}\n")
            best_config = best_by_seq[best_by_seq["seq_len"] == seq]
            if best_config.empty:
                f.write("  [No best configuration found]\n")
                continue
            baseline = df[(df["seq_len"] == seq) & (df["nvml_pwr_limit"] == 250)]
            if baseline.empty:
                f.write("  [No baseline (250W) configuration found]\n")
                continue
            baseline_row = baseline.iloc[0]
            best_row = best_config.iloc[0]
            energy_baseline = float(baseline_row["Joules/token"])
            energy_best = float(best_row["Joules/token"])
            abs_saving = energy_baseline - energy_best
            pct_saving = (abs_saving / energy_baseline) * 100 if energy_baseline > 0 else 0.0
            f.write("  Best configuration:\n")
            # Use scientific notation formatting for floating-point values.
            f.write(f"    nvml_pwr_limit: {int(best_row['nvml_pwr_limit'])} W\n")
            f.write(f"    seq_len: {int(best_row['seq_len'])}\n")
            f.write(f"    Block Dimensions: {int(best_row['block_size_x'])}x{int(best_row['block_size_y'])}\n")
            f.write(f"    Joules/token: {energy_best:.3e}\n")
            f.write(f"    Throughput: {float(best_row['throughput']):.2f} tokens/s\n")
            f.write(f"  Baseline at 250W: Joules/token = {energy_baseline:.3e}\n")
            f.write(f"  Energy saved: {abs_saving:.3e} Joules/token ({pct_saving:.1f}% reduction)\n\n")
    print(f"[INFO] Analysis report saved to: {report_file}")
    
    agg_csv = os.path.join(out_dir, "aggregated_stats_rq2.csv")
    agg_stats.to_csv(agg_csv, index=False)
    print(f"[INFO] Aggregated stats saved to: {agg_csv}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis_rq2.py <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    if not os.path.isfile(csv_file):
        print(f"[ERROR] CSV file not found: {csv_file}")
        sys.exit(1)
    
    out_dir = "rq2_plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Read the CSV data with high precision.
    df = pd.read_csv(csv_file, dtype={"Joules/token": np.float64})
    df = compute_throughput(df)
    
    # Find best configurations per (nvml_pwr_limit, seq_len)
    best_configs = find_best_configs(df)
    # Also, find the best config overall per sequence length.
    best_by_seq = find_best_config_by_seq(df)
    
    # Generate subplots for best configuration vs. power limit (one per sequence length).
    plot_best_config_vs_powerlimit_subplots(best_configs, out_dir=out_dir)
    
    # Generate the text report with aggregated statistics and best configs per sequence length.
    generate_text_report(df, best_configs, best_by_seq, out_dir=out_dir)
    
    print("[INFO] RQ2 analysis complete. See output in the 'rq2_plots' directory.")

if __name__ == "__main__":
    main()
