#!/usr/bin/env python3
import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_metrics(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert columns to numeric for both actual and predicted values.
    numeric_cols = ['DataSize', 'ThreadCount', 'BlockX', 'BlockY', 'GridX', 'GridY',
                    'EstTime(ns)', 'ActTime(ns)', 'PredictedPower(W)', 'ActualPower(W)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Create a column for block shape configuration (e.g., "1x32")
    df['BlockShape'] = df.apply(lambda row: f"{int(row['BlockX'])}x{int(row['BlockY'])}", axis=1)
    df['Config'] = df['BlockShape']  # Using block shape as the configuration label

    # Compute energy as the product of time and power (ns * W)
    df['EstEnergy'] = df['EstTime(ns)'] * df['PredictedPower(W)']
    df['ActEnergy'] = df['ActTime(ns)'] * df['ActualPower(W)']
    
    # Process each kernel separately
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].copy()
        # Sort by thread count and block shape for consistency.
        df_kernel = df_kernel.sort_values(by=['ThreadCount', 'BlockX', 'BlockY'])
        x = np.arange(len(df_kernel))
        config_labels = df_kernel['Config'].tolist()
        
        # Create a figure with three subplots: Time, Power, Energy.
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        ax1, ax2, ax3 = axes
        
        # Plot Time: Actual and Predicted
        ax1.plot(x, df_kernel['ActTime(ns)'], marker='o', linestyle='-', color='blue', label='Actual Time')
        ax1.plot(x, df_kernel['EstTime(ns)'], marker='s', linestyle='--', color='orange', label='Predicted Time')
        ax1.set_ylabel("Time (ns)")
        ax1.set_title(f"{kernel} - Combined Metrics vs. Block Configuration")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=10)
        
        # Plot Power: Actual and Predicted
        ax2.plot(x, df_kernel['ActualPower(W)'], marker='o', linestyle='-', color='green', label='Actual Power')
        ax2.plot(x, df_kernel['PredictedPower(W)'], marker='s', linestyle='--', color='red', label='Predicted Power')
        ax2.set_ylabel("Power (W)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=10)
        
        # Plot Energy: Actual and Predicted
        ax3.plot(x, df_kernel['ActEnergy'], marker='o', linestyle='-', color='purple', label='Actual Energy')
        ax3.plot(x, df_kernel['EstEnergy'], marker='s', linestyle='--', color='brown', label='Predicted Energy')
        ax3.set_ylabel("Energy (ns·W)")
        ax3.set_xlabel("Block Shape (BlockX x BlockY)")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(fontsize=10)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=10)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_combined_time_power_energy.png"
        plt.savefig(outname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved plot: {outname}")

def plot_actual_time_power_energy(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert columns to numeric
    for col in ['DataSize', 'ThreadCount', 'BlockX', 'BlockY', 'GridX', 'GridY',
                'ActTime(ns)', 'ActualPower(W)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing BlockX or BlockY
    df = df.dropna()

    # Create a column for block shape configuration (e.g., "1x32")
    df['BlockShape'] = df.apply(lambda row: f"{int(row['BlockX'])}x{int(row['BlockY'])}", axis=1)
    df['Config'] = df['BlockShape']  # Use block shape as configuration label

    # Compute actual energy (ns * W); units here are arbitrary
    df['ActualEnergy'] = df['ActTime(ns)'] * df['ActualPower(W)']

    # Process each kernel separately
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].copy()
        df_kernel = df_kernel.sort_values(by=['ThreadCount', 'BlockX', 'BlockY'])
        x = np.arange(len(df_kernel))
        config_labels = df_kernel['Config'].tolist()

        # Create a figure with three subplots: Time, Power, and Energy.
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot Actual Time (ns)
        ax1.plot(x, df_kernel['ActTime(ns)'], marker='o', linestyle='-', color='blue', label='Actual Time')
        ax1.set_ylabel("Actual Time (ns)")
        ax1.set_title(f"{kernel} - Actual Metrics vs. Block Configuration")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=10)

        # Plot Actual Power (W)
        ax2.plot(x, df_kernel['ActualPower(W)'], marker='o', linestyle='-', color='green', label='Actual Power')
        ax2.set_ylabel("Actual Power (W)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=10)

        # Plot Actual Energy (ns * W)
        ax3.plot(x, df_kernel['ActualEnergy'], marker='o', linestyle='-', color='red', label='Actual Energy')
        ax3.set_ylabel("Actual Energy (ns·W)")
        ax3.set_xlabel("Block Shape (BlockX x BlockY)")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(fontsize=10)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=10)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_actual_time_power_energy.png"
        plt.savefig(outname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved plot: {outname}")

def plot_predicted_time_power_energy(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert columns to numeric
    for col in ['DataSize', 'ThreadCount', 'BlockX', 'BlockY', 'GridX', 'GridY',
                'EstTime(ns)', 'PredictedPower(W)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()

    # Create a column for block shape configuration (e.g., "1x32")
    df['BlockShape'] = df.apply(lambda row: f"{int(row['BlockX'])}x{int(row['BlockY'])}", axis=1)
    df['Config'] = df['BlockShape']

    # Compute predicted energy (ns * W)
    df['PredictedEnergy'] = df['EstTime(ns)'] * df['PredictedPower(W)']

    # Process each kernel separately
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].copy()
        df_kernel = df_kernel.sort_values(by=['ThreadCount', 'BlockX', 'BlockY'])
        x = np.arange(len(df_kernel))
        config_labels = df_kernel['Config'].tolist()

        # Create a figure with three subplots: Time, Power, and Energy.
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot Predicted Time (ns)
        ax1.plot(x, df_kernel['EstTime(ns)'], marker='o', linestyle='-', color='blue', label='Predicted Time')
        ax1.set_ylabel("Predicted Time (ns)")
        ax1.set_title(f"{kernel} - Predicted Metrics vs. Block Configuration")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=10)

        # Plot Predicted Power (W)
        ax2.plot(x, df_kernel['PredictedPower(W)'], marker='o', linestyle='-', color='green', label='Predicted Power')
        ax2.set_ylabel("Predicted Power (W)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=10)

        # Plot Predicted Energy (ns * W)
        ax3.plot(x, df_kernel['PredictedEnergy'], marker='o', linestyle='-', color='red', label='Predicted Energy')
        ax3.set_ylabel("Predicted Energy (ns·W)")
        ax3.set_xlabel("Block Shape (BlockX x BlockY)")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(fontsize=10)

        ax3.set_xticks(x)
        ax3.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=10)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_predicted_time_power_energy.png"
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
    plot_actual_time_power_energy(csv_file)
    plot_predicted_time_power_energy(csv_file)
    plot_combined_metrics(csv_file)
    print("[INFO] Analysis complete.")

if __name__ == '__main__':
    main()
