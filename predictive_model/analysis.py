#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_times(csv_file):
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(csv_file)
    
    # Convert columns to numeric.
    df['DataSize'] = pd.to_numeric(df['DataSize'], errors='coerce')
    df['EstimatedTime(ns)'] = pd.to_numeric(df['EstimatedTime(ns)'], errors='coerce')
    df['ActualTime(ns)'] = pd.to_numeric(df['ActualTime(ns)'], errors='coerce')
    df['Difference(%)'] = pd.to_numeric(df['Difference(%)'], errors='coerce')
    
    # Create a new column "Config" representing the parameter combination.
    df['Config'] = df.apply(lambda row: f"D{int(row['DataSize'])}-G{row['GridX']}-B{row['BlockX']}", axis=1)
    
    # Process each kernel separately.
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].reset_index(drop=True)
        num_points = len(df_kernel)
        # X-axis: simple point index d1, d2, ...
        x = np.arange(1, num_points + 1)
        x_labels = [f"d{i}" for i in x]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Actual and Estimated times.
        ax.plot(x, df_kernel['ActualTime(ns)'], marker='o', linestyle='-', color='blue')
        ax.plot(x, df_kernel['EstimatedTime(ns)'], marker='s', linestyle='--', color='orange')
        
        ax.set_xlabel("Datapoint")
        ax.set_ylabel("Time (ns)")
        ax.set_title(f"{kernel} - Execution Times")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add a text block listing configurations for each datapoint.
        config_text = "\n".join([f"d{i}: {row['Config']}" for i, row in df_kernel.iterrows()])
        plt.figtext(
            0.02, 0.02, config_text, horizontalalignment="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.5)
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_time_lineplot.png"
        plt.savefig(outname)
        plt.close()
        print(f"[INFO] Saved time plot: {outname}")

def plot_power(csv_file):
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(csv_file)
    
    # Convert power columns to numeric.
    df['EstimatedPower(W)'] = pd.to_numeric(df['EstimatedPower(W)'], errors='coerce')
    df['ActualPower(W)'] = pd.to_numeric(df['ActualPower(W)'], errors='coerce')
    
    # Create a new column "Config" representing the parameter combination.
    df['Config'] = df.apply(lambda row: f"D{int(row['DataSize'])}-G{row['GridX']}-B{row['BlockX']}", axis=1)
    
    # Process each kernel separately.
    kernels = df['Kernel'].unique()
    for kernel in kernels:
        df_kernel = df[df['Kernel'] == kernel].reset_index(drop=True)
        num_points = len(df_kernel)
        x = np.arange(1, num_points + 1)
        x_labels = [f"d{i}" for i in x]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, df_kernel['ActualPower(W)'], marker='o', linestyle='-', color='green')
        ax.plot(x, df_kernel['EstimatedPower(W)'], marker='s', linestyle='--', color='red')
        
        ax.set_xlabel("Datapoint")
        ax.set_ylabel("Power (W)")
        ax.set_title(f"{kernel} - Power Estimates")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        config_text = "\n".join([f"d{i}: {row['Config']}" for i, row in df_kernel.iterrows()])
        plt.figtext(
            0.02, 0.02, config_text, horizontalalignment="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.5)
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        os.makedirs("plots", exist_ok=True)
        outname = f"plots/{kernel}_power_lineplot.png"
        plt.savefig(outname)
        plt.close()
        print(f"[INFO] Saved power plot: {outname}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis.py <results.csv>")
        sys.exit(1)
    csv_file = sys.argv[1]
    plot_times(csv_file)
    plot_power(csv_file)

if __name__ == '__main__':
    main()
