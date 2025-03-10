import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from matplotlib.gridspec import GridSpec

def sort_thread_configs(configs):
    def config_key(config):
        x, y = eval(config.replace('(', '').replace(')', ''))
        return (x * y, x)
    return sorted(configs, key=config_key)

def process_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert energy from μJ to mJ and time from μs to ms
    df['energy_mJ'] = df['energy_uj'] / 1000.0
    df['time_ms'] = df['duration_us'] / 1000.0
    
    # Create thread configuration and total threads
    df['thread_config'] = '(' + df['block_x'].astype(str) + ',' + df['block_y'].astype(str) + ')'
    df['total_threads'] = df['block_x'] * df['block_y']
    
    # Extract matrix size from filename
    filename = os.path.basename(file_path)
    size_part = filename.split('_')[2]
    n, d = map(int, size_part.split('x'))
    df['n'] = n
    df['d'] = d
    
    return df

def plot_individual_kernel_metrics(data_files):
    # Get all unique kernel names across all files
    all_kernels = set()
    for file_path in data_files:
        df = pd.read_csv(file_path)
        all_kernels.update(df['kernel_name'].unique())
    
    # Style configuration for matrix sizes
    matrix_styles = {
        '1024x1024': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        '4096x4096': {'color': 'red', 'marker': 's', 'linestyle': '--'},
        '8192x8192': {'color': 'green', 'marker': '^', 'linestyle': '-.'},
        '16384x16384': {'color': 'purple', 'marker': 'D', 'linestyle': ':'}
    }

    # Create plots for each kernel
    for kernel in all_kernels:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        fig.suptitle(f'Performance Metrics for {kernel}', fontsize=16)
        
        all_configs = set()
        data_by_size = {}

        for file_path in data_files:
            # Extract matrix size from filename
            parts = os.path.basename(file_path).split('_')
            matrix_size = f"{parts[2]}x{parts[3]}"
            
            df = process_data(file_path)
            kernel_data = df[df['kernel_name'] == kernel]
            
            if not kernel_data.empty:
                # Group by thread configuration
                grouped = kernel_data.groupby('thread_config').agg({
                    'energy_mJ': 'mean',
                    'time_ms': 'mean',
                    'power_mW': 'mean',
                    'block_x': 'first',
                    'block_y': 'first',
                    'total_threads': 'first'
                }).reset_index()
                
                all_configs.update(grouped['thread_config'])
                data_by_size[matrix_size] = grouped

        if not data_by_size:
            continue  # Skip if no data for this kernel

        sorted_configs = sort_thread_configs(list(all_configs))

        # Plot each matrix size
        for matrix_size, data in data_by_size.items():
            style = matrix_styles.get(matrix_size, {})
            data = data.set_index('thread_config').reindex(sorted_configs).reset_index()
            
            ax1.plot(data['thread_config'], data['energy_mJ'], 
                    label=matrix_size, **style)
            ax2.plot(data['thread_config'], data['time_ms'],
                    label=matrix_size, **style)
            ax3.plot(data['thread_config'], data['power_mW'],
                    label=matrix_size, **style)

        # Configure axes
        ax1.set_ylabel('Energy (mJ)', fontsize=12)
        ax2.set_ylabel('Execution Time (ms)', fontsize=12)
        ax3.set_ylabel('Power (mW)', fontsize=12)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('Thread Block Configuration (x,y)', fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(title='Matrix Size', fontsize=10)

        plt.tight_layout()
        plt.savefig(f'performance_metrics_{kernel}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_metrics(data_files):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 15))
    fig.suptitle('Performance Metrics Analysis', fontsize=16)

    # Style configuration
    kernel_styles = {
        'kernel1': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'kernel2': {'color': 'green', 'marker': 's', 'linestyle': '--'},
        'kernel3': {'color': 'red', 'marker': '^', 'linestyle': ':'}
    }

    matrix_sizes = set()
    all_configs = set()
    
    # Collect and process data
    for file_path in data_files:
        matrix_size = f"{file_path.split('_')[2]}x{file_path.split('_')[3].split('_')[0]}"
        matrix_sizes.add(matrix_size)
        grouped = process_data(file_path)
        all_configs.update(grouped['thread_config'])

    sorted_configs = sort_thread_configs(list(all_configs))
    
    # Plotting
    for file_path in data_files:
        matrix_size = f"{file_path.split('_')[2]}x{file_path.split('_')[3].split('_')[0]}"
        data = process_data(file_path)
        
        for kernel, kdata in data.groupby('kernel_name'):
            kdata = kdata.set_index('thread_config').reindex(sorted_configs).reset_index()
            style = kernel_styles.get(kernel, {})
            
            ax1.plot(kdata['thread_config'], kdata['energy_mJ'], 
                    label=f'{matrix_size} - {kernel}', **style)
            ax2.plot(kdata['thread_config'], kdata['time_ms'],
                    label=f'{matrix_size} - {kernel}', **style)
            ax3.plot(kdata['thread_config'], kdata['power_mW'],
                    label=f'{matrix_size} - {kernel}', **style)

    # Plot configuration
    ax1.set_ylabel('Energy (mJ)', fontsize=12)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax3.set_ylabel('Power (mW)', fontsize=12)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Thread Block Configuration (x,y)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig('detailed_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_tradeoffs(data_files):
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])  # Energy-Time
    ax2 = fig.add_subplot(gs[0, 1])  # Power-Time
    ax3 = fig.add_subplot(gs[1, :])  # Efficiency

    kernel_markers = {
        'kernel1': 'o',
        'kernel2': 's',
        'kernel3': '^'
    }

    for file_path in data_files:
        matrix_size = f"{file_path.split('_')[2]}x{file_path.split('_')[3].split('_')[0]}"
        data = process_data(file_path)
        
        for kernel, kdata in data.groupby('kernel_name'):
            # Energy-Time Tradeoff
            ax1.scatter(kdata['time_ms'], kdata['energy_mJ'], 
                       marker=kernel_markers.get(kernel, 'o'),
                       label=f'{matrix_size} - {kernel}')
            
            # Power-Time Relationship
            ax2.scatter(kdata['time_ms'], kdata['power_mW'],
                       marker=kernel_markers.get(kernel, 'o'),
                       label=f'{matrix_size} - {kernel}')
            
            # Efficiency Calculation using actual matrix dimensions from data
            kdata['efficiency'] = (kdata['n'] * kdata['d']) / (kdata['time_ms'] * kdata['energy_mJ'])
            ax3.plot(kdata['thread_config'], kdata['efficiency'],
                    marker=kernel_markers.get(kernel, 'o'),
                    label=f'{matrix_size} - {kernel}')

    # Axis labels and formatting
    ax1.set_title('Energy-Time Tradeoff')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Energy (mJ)')
    
    ax2.set_title('Power-Time Relationship')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Power (mW)')
    
    ax3.set_title('Computational Efficiency')
    ax3.set_xlabel('Thread Block Configuration')
    ax3.set_ylabel('Elements/(ms·mJ)')
    ax3.tick_params(axis='x', rotation=45)
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig('tradeoff_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data_files = glob.glob('power_data_*x*_impl*.csv')
    data_files.sort(key=lambda x: int(x.split('_')[2].split('x')[0]))
    
    plot_individual_kernel_metrics(data_files)
    plot_tradeoffs(data_files)