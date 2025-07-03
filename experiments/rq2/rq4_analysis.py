#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_times(csv_file):
    df = pd.read_csv(csv_file)
    
    # Convert numeric columns
    df['DataSize'] = pd.to_numeric(df['DataSize'], errors='coerce')
    df['ThreadCount'] = pd.to_numeric(df['ThreadCount'], errors='coerce')
    df['BlockX'] = pd.to_numeric(df['BlockX'], errors='coerce')
    df['BlockY'] = pd.to_numeric(df['BlockY'], errors='coerce')
    df['GridX'] = pd.to_numeric(df['GridX'], errors='coerce')
    df['GridY'] = pd.to_numeric(df['GridY'], errors='coerce')
    df['EstTime(ns)'] = pd.to_numeric(df['EstTime(ns)'], errors='coerce')
    df['ActTime(ns)'] = pd.to_numeric(df['ActTime(ns)'], errors='coerce')
    df['DiffTime(%)'] = pd.to_numeric(df['DiffTime(%)'], errors='coerce')
    
    # Create columns for block shape and grid shape
    df['BlockShape'] = df.apply(lambda row: f"{int(row['BlockX'])}x{int(row['BlockY'])}", axis=1)
    df['GridShape'] = df.apply(lambda row: f"{int(row['GridX'])}x{int(row['GridY'])}", axis=1)
    
    # Create a sort key for block shape: (ThreadCount, BlockX, BlockY)
    df['sort_key'] = df.apply(lambda row: (row['ThreadCount'], row['BlockX'], row['BlockY']), axis=1)
    df = df.sort_values(by=['ThreadCount', 'BlockX', 'BlockY'])
    df['Config'] = df.apply(lambda row: f"{row['BlockShape']}", axis=1)
    
    # For each kernel, and for each combination of DataSize and GridShape, create a subplot.
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].copy()
        data_sizes = sorted(df_kernel['DataSize'].unique())
        grid_shapes = sorted(df_kernel['GridShape'].unique())
        
        # Create a subplot grid: rows = number of data sizes, columns = number of grid shapes.
        fig, axes = plt.subplots(len(data_sizes), len(grid_shapes),
                                 figsize=(6*len(grid_shapes), 4*len(data_sizes)),
                                 squeeze=False, sharex=True)
        
        for i, ds in enumerate(data_sizes):
            for j, gs in enumerate(grid_shapes):
                subdf = df_kernel[(df_kernel['DataSize'] == ds) & (df_kernel['GridShape'] == gs)]
                if subdf.empty:
                    axes[i, j].set_visible(False)
                    continue
                # Ensure sorted order by our sort_key.
                subdf = subdf.sort_values(by=['sort_key'])
                
                x = np.arange(len(subdf))
                xlabels = subdf['Config'].tolist()
                
                # Plot Predicted Time with circular markers, solid line, purple color.
                axes[i, j].plot(
                    x,
                    subdf['EstTime(ns)'],
                    marker='o', linestyle='-', color='purple',
                    label='Predicted Time'
                )
                # Plot Actual Time with circular markers, solid line, red color.
                axes[i, j].plot(
                    x,
                    subdf['ActTime(ns)'],
                    marker='o', linestyle='-', color='red',
                    label='Actual Time'
                )
                
                axes[i, j].set_title(f"DataSize: {ds} | Grid: {gs}", fontsize=10)
                axes[i, j].set_ylabel("Time (ns)", fontsize=10)
                axes[i, j].set_xticks(x)
                axes[i, j].set_xticklabels(xlabels, rotation=45, ha='right', fontsize=10)
                axes[i, j].legend(fontsize=10)
                axes[i, j].grid(True, linestyle='--', alpha=0.5)
        
        fig.suptitle(f"Time Comparison - {kernel}", fontsize=16, y=0.98)
        fig.text(0.5, 0.04, "Block Shape (Sorted by ThreadCount)", ha='center', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_time.png"
        plt.savefig(outname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved time plot: {outname}")

def plot_power(csv_file):
    df = pd.read_csv(csv_file)
    
    df['ThreadCount'] = pd.to_numeric(df['ThreadCount'], errors='coerce')
    df['BlockX'] = pd.to_numeric(df['BlockX'], errors='coerce')
    df['BlockY'] = pd.to_numeric(df['BlockY'], errors='coerce')
    df['GridX'] = pd.to_numeric(df['GridX'], errors='coerce')
    df['GridY'] = pd.to_numeric(df['GridY'], errors='coerce')
    df['DataSize'] = pd.to_numeric(df['DataSize'], errors='coerce')
    df['PredictedPower(W)'] = pd.to_numeric(df['PredictedPower(W)'], errors='coerce')
    df['ActualPower(W)'] = pd.to_numeric(df['ActualPower(W)'], errors='coerce')
    
    df['BlockShape'] = df.apply(lambda row: f"{int(row['BlockX'])}x{int(row['BlockY'])}", axis=1)
    df['GridShape'] = df.apply(lambda row: f"{int(row['GridX'])}x{int(row['GridY'])}", axis=1)
    df['sort_key'] = df.apply(lambda row: (row['ThreadCount'], row['BlockX'], row['BlockY']), axis=1)
    df = df.sort_values(by=['ThreadCount', 'BlockX', 'BlockY'])
    df['Config'] = df.apply(lambda row: f"{row['BlockShape']}", axis=1)
    
    # Group by Kernel, then by DataSize and GridShape
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].copy()
        data_sizes = sorted(df_kernel['DataSize'].unique())
        grid_shapes = sorted(df_kernel['GridShape'].unique())
        
        fig, axes = plt.subplots(len(data_sizes), len(grid_shapes),
                                 figsize=(6*len(grid_shapes), 4*len(data_sizes)),
                                 squeeze=False, sharex=True)
        
        for i, ds in enumerate(data_sizes):
            for j, gs in enumerate(grid_shapes):
                subdf = df_kernel[(df_kernel['DataSize'] == ds) & (df_kernel['GridShape'] == gs)]
                if subdf.empty:
                    axes[i, j].set_visible(False)
                    continue
                subdf = subdf.sort_values(by=['sort_key'])
                
                x = np.arange(len(subdf))
                xlabels = subdf['Config'].tolist()
                
                # Plot Predicted Power with circular markers, solid line, purple color.
                axes[i, j].plot(
                    x,
                    subdf['PredictedPower(W)'],
                    marker='o', linestyle='-', color='purple',
                    label='Predicted Power'
                )
                # Plot Actual Power with circular markers, solid line, red color.
                axes[i, j].plot(
                    x,
                    subdf['ActualPower(W)'],
                    marker='o', linestyle='-', color='red',
                    label='Actual Power'
                )
                
                axes[i, j].set_title(f"DataSize: {ds} | Grid: {gs}", fontsize=10)
                axes[i, j].set_ylabel("Power (W)", fontsize=10)
                axes[i, j].set_xticks(x)
                axes[i, j].set_xticklabels(xlabels, rotation=45, ha='right', fontsize=10)
                axes[i, j].legend(fontsize=10)
                axes[i, j].grid(True, linestyle='--', alpha=0.5)
        
        fig.suptitle(f"Power Consumption - {kernel}", fontsize=16, y=0.98)
        fig.text(0.5, 0.04, "Block Shape (Sorted by ThreadCount)", ha='center', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_power.png"
        plt.savefig(outname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved power plot: {outname}")

def find_latest_csv(default_dir="experiment-results"):
    csv_files = glob.glob(os.path.join(default_dir, "results-*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files found in '{default_dir}'!")
        sys.exit(1)
    csv_files.sort()  # Filenames are sortable thanks to the timestamp.
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
    plot_times(csv_file)
    plot_power(csv_file)
    print("[INFO] Analysis complete.")

if __name__ == '__main__':
    main()
