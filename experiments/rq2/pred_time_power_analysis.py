#!/usr/bin/env python3
import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_Predicted_time_and_power(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert columns to numeric
    for col in ['DataSize', 'ThreadCount', 'BlockX', 'BlockY', 'GridX', 'GridY',
                'EstTime(ns)', 'PredictedPower(W)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create a column for the block shape configuration (e.g., "1x32")
    df['BlockShape'] = df.apply(lambda row: f"{int(row['BlockX'])}x{int(row['BlockY'])}", axis=1)
    
    # Optionally create a label combining grid shape if desired:
    # df['Config'] = df.apply(lambda row: f"{row['BlockShape']} | Grid: {int(row['GridX'])}x{int(row['GridY'])}", axis=1)
    df['Config'] = df['BlockShape']  # For simplicity, use block shape as the configuration label

    # Process each kernel separately
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].copy()
        # Sort by thread count and block shape for consistency
        df_kernel = df_kernel.sort_values(by=['ThreadCount', 'BlockX', 'BlockY'])
        
        x = np.arange(len(df_kernel))
        config_labels = df_kernel['Config'].tolist()

        # Create a figure with two subplots (Predicted Time and Predicted Power)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot Predicted Time (ns)
        ax1.plot(x, df_kernel['EstTime(ns)'], marker='o', linestyle='-', color='blue', label='Predicted Time')
        ax1.set_ylabel("Predicted Time (ns)")
        ax1.set_title(f"{kernel} - Predicted Time vs. Block Configuration")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=10)

        # Plot Predicted Power (W)
        ax2.plot(x, df_kernel['PredictedPower(W)'], marker='o', linestyle='-', color='green', label='Predicted Power')
        ax2.set_ylabel("Predicted Power (W)")
        ax2.set_xlabel("Block Shape (BlockX x BlockY)")
        ax2.set_title(f"{kernel} - Predicted Power vs. Block Configuration")
        ax2.set_xticks(x)
        ax2.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=10)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_predicted_time_power.png"
        plt.savefig(outname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved plot: {outname}")

def find_latest_csv(default_dir="experiment-results"):
    csv_files = glob.glob(os.path.join(default_dir, "results-*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in '{default_dir}'!")
        sys.exit(1)
    csv_files.sort()  # Assumes filenames include a sortable timestamp.
    return csv_files[-1]

def main():
    if len(sys.argv) == 2:
        csv_file = sys.argv[1]
    else:
        csv_file = find_latest_csv()

    if not os.path.exists(csv_file):
        print(f"[ERROR] CSV file not found: {csv_file}")
        sys.exit(1)
    
    print(f"[INFO] Analyzing CSV file: {csv_file}")
    plot_Predicted_time_and_power(csv_file)
    print("[INFO] Analysis complete.")

if __name__ == '__main__':
    main()
