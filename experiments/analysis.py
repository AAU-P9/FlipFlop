import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from scipy.spatial import ConvexHull

class GEMMEnergyAnalyzer:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(data)
        
        self.device = data['device_name']
        self.problem_size = data['problem_size']
        self.df = self._parse_json_data(data)
        self.output_dir = f"analysis/{os.path.basename(json_path).split('.')[0]}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-paper')
        self._set_plot_style()

    def _set_plot_style(self):
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'figure.figsize': (10, 6),
            'lines.linewidth': 2,
            'lines.markersize': 8
        })

    def _parse_json_data(self, data):
        rows = []
        M, N = self.problem_size
        total_flops = 2 * M * N * M  # For GEMM operation
        
        for config, metrics in data['cache'].items():
            row = {
                'block_x': metrics['block_size_x'],
                'block_y': metrics['block_size_y'],
                'tile_x': metrics['tile_size_x'],
                'tile_y': metrics['tile_size_y'],
                'time_ms': metrics['time'],
                'power_w': metrics['nvml_power'],
                'nvml_pwr_limit': metrics['nvml_pwr_limit'],
                'nvml_energy': metrics['nvml_energy'],
                'SM_Occupancy': metrics.get('sm__warps_active.avg.pct_of_peak_sustained_active', 0),
                'flops_watt': total_flops / metrics['nvml_energy'],
                'edp': metrics['edp'],
                'block_dims': f"{metrics['block_size_x']}x{metrics['block_size_y']}",
                'thread_count': metrics['block_size_x'] * metrics['block_size_y'],
            }
            rows.append(row)
            
        return pd.DataFrame(rows)

    def plot_power_adaptive_frontiers(self):
        """Dual-line plot of SM occupancy and energy efficiency"""
        # Data preparation and sorting
        self.df['block_dims'] = self.df.apply(
            lambda r: f"{int(r['block_x'])}x{int(r['block_y'])}", axis=1
        )
        self.df['thread_count'] = self.df['block_x'] * self.df['block_y']
        self.df['sort_key'] = self.df.apply(
            lambda r: (r['thread_count'], r['block_x'], r['block_y']), axis=1
        )
        
        block_order = (self.df[['block_dims', 'sort_key']]
                    .drop_duplicates()
                    .sort_values('sort_key')
                    ['block_dims'].tolist())
        
        self.df['block_dims'] = pd.Categorical(
            self.df['block_dims'],
            categories=block_order,
            ordered=True
        )
        self.df = self.df.sort_values('block_dims')
        self.df['block_code'] = self.df['block_dims'].cat.codes

        # Create figure
        fig, ax1 = plt.subplots(figsize=(14, 7))
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11
        })

        # SM Occupancy line (Royal Blue)
        sm_line = ax1.plot(
            self.df['block_code'].unique(),
            self.df.groupby('block_code')['SM_Occupancy'].mean(),
            color='#4169E1',
            marker='o',
            markersize=8,
            linewidth=2.5,
            label='SM Occupancy',
            zorder=3
        )
        ax1.set_ylabel('SM Occupancy (%)', color='#4169E1', fontsize=15)
        ax1.tick_params(axis='y', labelcolor='#4169E1', labelsize=14)
        ax1.set_ylim(0, 100)

        # Energy line (Crimson Red)
        ax2 = ax1.twinx()
        energy_line = ax2.plot(
            self.df['block_code'].unique(),
            self.df.groupby('block_code')['nvml_energy'].mean(),
            color='#DC143C',
            marker='s',
            markersize=8,
            linewidth=2.5,
            label='Energy Consumption',
            zorder=2
        )
        ax2.set_ylabel('Energy (J)', color='#DC143C')
        ax2.tick_params(axis='y', labelcolor='#DC143C')

        # X-axis configuration
        ax1.set_xticks(self.df['block_code'].unique())
        ax1.set_xticklabels(
            [d.replace('x', '×') for d in block_order],
            rotation=45,
            ha='right',
            fontsize=13
        )
        ax1.set_xlabel('Block Dimensions (X×Y)', labelpad=15, fontsize=15)

        # Annotation and legend
        min_energy_idx = self.df['nvml_energy'].idxmin()
        ax2.annotate('Optimal Config',
                    xy=(self.df.loc[min_energy_idx, 'block_code'], 
                        self.df.loc[min_energy_idx, 'nvml_energy']),
                    xytext=(15, -40),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5),
                    fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=1))

        lines = sm_line + energy_line
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True,
                framealpha=0.9, facecolor='white')

        plt.title('SM Utilization and Energy Efficiency by Block Configuration', pad=20, fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sm_energy_lineplot.png", dpi=300, bbox_inches='tight')
        plt.close()


    def plot_energy_adaptation(self):
        """Single plot showing energy vs block dimensions across power limits"""
        # Prepare data with proper ordering
        self.df['block_dims'] = self.df.apply(
            lambda r: f"{int(r['block_x'])}x{int(r['block_y'])}", axis=1
        )
        
        # Create sorting keys like in configuration analysis
        self.df['thread_count'] = self.df['block_x'] * self.df['block_y']
        self.df['sort_key'] = self.df.apply(
            lambda r: (r['thread_count'], r['block_x'], r['block_y']), axis=1
        )
        
        # Get sorted block order based on sort_key
        block_order = (self.df[['block_dims', 'sort_key']]
                    .drop_duplicates()
                    .sort_values('sort_key')
                    ['block_dims'].tolist())
        
        # Create categorical type with correct ordering
        self.df['block_code'] = self.df['block_dims'].astype(
            pd.CategoricalDtype(categories=block_order, ordered=True)
        ).cat.codes

        # Explicitly create figure and axes
        fig, ax = plt.subplots(figsize=(15, 5))
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14
        })

        # Create colormap for power limits
        power_limits = sorted(self.df['nvml_pwr_limit'].unique())
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(min(power_limits), max(power_limits))

        # Plot lines for each power limit
        for pwr in power_limits:
            subset = self.df[self.df['nvml_pwr_limit'] == pwr]
            if not subset.empty:
                # Sort by block_code to maintain ordering
                grouped = subset.groupby('block_code', observed=True).agg({
                    'nvml_energy': 'mean',
                    'block_dims': 'first'
                }).sort_index()
                
                ax.plot(
                    grouped.index,
                    grouped['nvml_energy'],
                    color=cmap(norm(pwr)),
                    linewidth=2,
                    marker='o',
                    markersize=8,
                    label=f'{pwr}W'
                )

        # Configure axis and labels with consistent ordering
        ax.set_xticks(grouped.index)
        ax.set_xticklabels(
            [b.replace('x', '×') for b in block_order],  # Use pre-sorted block_order
            rotation=45,
            ha='right',
            fontsize=12,
        )
        ax.set_xlabel('Block Dimensions (X×Y)', labelpad=15, fontsize=16)
        ax.set_ylabel('Energy Consumption (J)', labelpad=15, fontsize=16)
        ax.set_title('Energy vs Block Dimensions Across Power Limits', pad=20)

        # Add colorbar with explicit axes reference
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Power Limit (W)', rotation=270, labelpad=24, fontsize=16)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/power_limit_energy_analysis.png", dpi=300)
        plt.close()

    def plot_configuration_analysis(self):
        """Vertical line plots with legend in first subplot"""
        # Data preparation
        self.df['block_dims'] = self.df.apply(
            lambda r: f"{int(r['block_x'])}x{int(r['block_y'])}", axis=1
        )
        self.df['thread_count'] = self.df['block_x'] * self.df['block_y']
        self.df['sort_key'] = self.df.apply(
            lambda r: (r['thread_count'], r['block_x'], r['block_y']), axis=1
        )
        
        # Create sorted categorical ordering
        block_order = (self.df[['block_dims', 'sort_key']]
                    .drop_duplicates()
                    .sort_values('sort_key')
                    ['block_dims'].tolist())
        self.df['block_dims'] = pd.Categorical(
            self.df['block_dims'],
            categories=block_order,
            ordered=True
        )
        self.df['block_code'] = self.df['block_dims'].cat.codes

        # Create vertical subplots
        fig, axs = plt.subplots(3, 1, figsize=(18, 10))
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize':12
        })

        # Get unique thread counts and assign colors
        unique_threads = sorted(self.df['thread_count'].unique())
        palette = sns.color_palette("tab10", n_colors=len(unique_threads))
        color_map = dict(zip(unique_threads, palette))

        # Create legend elements
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
            label=str(tc),
            markerfacecolor=color_map[tc], 
            markersize=8,
            markeredgewidth=0.5
        ) for tc in unique_threads]

        metrics = [
            ('nvml_energy', 'Energy (J)', 'Energy Consumption'),
            ('power_w', 'Power (W)', 'Power Draw'),
            ('time_ms', 'Time (ms)', 'Execution Time')
        ]

        for idx, (metric, ylabel, title) in enumerate(metrics):
            # Group and sort data
            grouped = self.df.groupby('block_dims', observed=True).agg(
                mean_value=(metric, 'mean'),
                block_code=('block_code', 'first'),
                thread_count=('thread_count', 'first')
            ).sort_values('block_code')

            # Plot lines and markers
            for i, (_, row) in enumerate(grouped.iterrows()):
                axs[idx].scatter(
                    row['block_code'],
                    row['mean_value'],
                    color=color_map[row['thread_count']],
                    s=60,
                    edgecolor='white',
                    zorder=3
                )
                
                if i < len(grouped)-1:
                    next_row = grouped.iloc[i+1]
                    axs[idx].plot(
                        [row['block_code'], next_row['block_code']],
                        [row['mean_value'], next_row['mean_value']],
                        color=color_map[row['thread_count']],
                        linewidth=1.5,
                        zorder=2
                    )

            # Configure axes
            axs[idx].set_title(title, pad=8)
            axs[idx].set_ylabel(ylabel, fontsize=14)
            axs[idx].tick_params(axis='y', labelsize=12)
            axs[idx].grid(True, linestyle='--', alpha=0.3)
            
            # Add legend to first subplot
            if idx == 0:
                axs[idx].legend(
                    handles=legend_elements,
                    title='Threads',
                    loc='upper right',
                    bbox_to_anchor=(1, 1),
                    ncol=3,
                    fontsize=12,
                    title_fontsize=14,
                    frameon=False,
                    handletextpad=0.1,
                    columnspacing=0.5
                )

            if idx == 2:
                axs[idx].set_xticks(grouped['block_code'])
                axs[idx].set_xticklabels(
                    [f"{b.replace('x', '×')}" for b in grouped.index],
                    rotation=45,
                    ha='right',
                    # ha='center',
                    fontsize=13,
                    fontfamily='DejaVu Sans'
                )
                axs[idx].set_xlabel('Block Dimensions\n(X × Y)', fontsize=14)
            else:
                axs[idx].set_xticks([])

        plt.tight_layout(pad=2, h_pad=1.5)
        fig.savefig(f"{self.output_dir}/vertical_block_analysis.png", 
                bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_time_vs_energy(self):
        """Scatter plot of time vs energy with power limit coloring"""
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(
            data=self.df,
            x='time_ms',
            y='nvml_energy',
            hue='nvml_pwr_limit',
            palette='viridis',
            sizes=(20, 200),
            alpha=0.7
        )
        plt.title('Execution Time vs Energy Consumption')
        plt.xlabel('Execution Time (ms)')
        plt.ylabel('Energy (J)')
        
        # Create custom legends
        handles, labels = scatter.get_legend_handles_labels()
        plt.legend(
            handles=handles,
            title='Power Limit/SM Occ.',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/time_vs_energy.png")
        plt.close()

    # Suggested plot additions for RQ1
    def plot_edp_heatmap(self):
        """EDP heatmap across block dimensions"""
        pivot = self.df.pivot_table(
            index='block_x', 
            columns='block_y', 
            values='edp', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot, 
            annot=False, 
            cmap='viridis',
            cbar_kws={'label': 'Energy-Delay Product (J·s²)'}
        )
        plt.title('EDP Distribution Across Block Dimensions')
        plt.xlabel('Block Y Dimension')
        plt.ylabel('Block X Dimension')
        plt.savefig(f"{self.output_dir}/edp_heatmap.png", bbox_inches='tight')
        plt.close()

    def _epsilon_filter(self, df, metrics, epsilon=0.1):
        """Epsilon-dominance filter for Pareto optimality"""
        df = df.sort_values(metrics[0]).reset_index(drop=True)  # Reset index
        mask = np.ones(len(df), dtype=bool)
        frontier = []
        
        for idx, row in df.iterrows():
            dominated = False
            for f in frontier:
                if all(row[m] >= (1 - epsilon)*f[m] for m in metrics):
                    dominated = True
                    break
            if not dominated:
                frontier.append(row)
                mask[idx] = True  # Use local index
            else:
                mask[idx] = False
        return mask

    def plot_pareto_front(self):
        """Pareto frontier visualization"""
        pareto_mask = self._epsilon_filter(self.df, ['edp', 'flops_watt'])
        pareto_df = self.df[pareto_mask]
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.df,
            x='edp',
            y='flops_watt',
            hue='thread_count',
            palette='viridis',
            alpha=0.6
        )
        sns.scatterplot(
            data=pareto_df,
            x='edp',
            y='flops_watt',
            color='red',
            s=100,
            label='Pareto Frontier'
        )
        plt.title('Pareto Frontier: EDP vs FLOPs/Watt')
        plt.xlabel('Energy-Delay Product (J·s²)')
        plt.ylabel('FLOPs per Watt')
        plt.legend()
        plt.savefig(f"{self.output_dir}/pareto_front.png", bbox_inches='tight')
        plt.close()

    def plot_3d_pareto(self):
        """Interactive 3D plot of Energy, Time, FLOPs/Watt"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        sc = ax.scatter(
            self.df['time_ms'], 
            self.df['nvml_energy'], 
            self.df['flops_watt'],
            c=self.df['edp'],
            cmap='viridis'
        )
        
        ax.set_xlabel('Time (ms)', labelpad=12)
        ax.set_ylabel('Energy (J)', labelpad=12)
        ax.set_zlabel('FLOPs/Watt', labelpad=12)
        fig.colorbar(sc, label='EDP (J·s²)', pad=0.1)
        
        plt.savefig(f"{self.output_dir}/3d_pareto.png")


    def plot_pareto(self):
        """Scatter plot of Time vs Energy with Power Limit color mapping"""
        plt.figure(figsize=(12, 8))
        
        # Create the main scatter plot
        scatter = plt.scatter(
            self.df['time_ms'], 
            self.df['nvml_energy'],
            c=self.df['nvml_pwr_limit'],
            cmap='viridis',
            alpha=0.7,
            s=40,
            edgecolors='w',
            linewidth=0.5
        )

        # Add color bar and labels
        cbar = plt.colorbar(scatter)
        cbar.set_label('Power Limit (W)', rotation=270, labelpad=15)
        
        plt.xlabel('Execution Time (ms)', fontsize=12)
        plt.ylabel('Energy Consumption (J)', fontsize=12)
        plt.title('Energy-Time Tradeoff with Power Limit Coloring', fontsize=14)
        
        # Add grid and styling
        plt.grid(True, alpha=0.2, linestyle='--')
        plt.gca().set_facecolor('#f5f5f5')
        
        # Save high-quality version
        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/time_energy_scatter.png", 
            dpi=300, 
            bbox_inches='tight'
        )
        plt.close()


    def plot_adaptation_timeline(self):
        """Energy consumption timeline with adaptation points"""
        # Requires time-series data from dynamic runs
        timeline = self._load_adaptation_logs()  # Implement based on actual data
        
        plt.figure(figsize=(14, 6))
        plt.plot(timeline['timestamp'], timeline['energy'], label='Adaptive')
        plt.plot(timeline['timestamp'], timeline['static_energy'], label='Static')
        plt.scatter(
            timeline[timeline['adapt_event']]['timestamp'],
            timeline[timeline['adapt_event']]['energy'],
            color='red',
            label='Configuration Update'
        )
        plt.title('Dynamic Adaptation Timeline')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (J)')
        plt.legend()
        plt.savefig(f"{self.output_dir}/adaptation_timeline.png")
        plt.close()
    
    def plot_bayesian_convergence(self):
        """Plot optimization progress over iterations"""
        plt.figure(figsize=(10,6))
        plt.plot(self.df['fevals'], self.df['objective'], 'b-')
        plt.xlabel('Function Evaluations')
        plt.ylabel('Objective (EDP + Penalty)')
        plt.title('Bayesian Optimization Convergence')
        plt.savefig("bayesian_convergence.png")

    def plot_configuration_space(self):
        """3D plot of block dims vs power vs EDP"""
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(
            self.df['block_x'], 
            self.df['block_y'], 
            self.df['nvml_pwr_limit'],
            c=self.df['edp'],
            cmap='viridis'
        )
        ax.set_xlabel('Block X')
        ax.set_ylabel('Block Y')
        ax.set_zlabel('Power Limit')
        plt.savefig("config_space_3d.png")

    


if __name__ == "__main__":
    analyzer = GEMMEnergyAnalyzer("/home/srajput/flipflop/cuda_kernel_energy_empirical/experiments/data/run2/power_tuning_float16.json")
    # analyzer.plot_power_adaptive_frontiers()
    # analyzer.plot_energy_adaptation()
    # analyzer.plot_configuration_analysis()
    # analyzer.plot_time_vs_energy()

    # Additional plots for RQ1
    # analyzer.plot_edp_heatmap()
    # analyzer.plot_pareto_front()
    # analyzer.plot_energy_adaptation()
    analyzer.plot_configuration_analysis()
    # analyzer.plot_3d_pareto()
    analyzer.plot_pareto()
    # analyzer.plot_bayesian_convergence()
    # analyzer.plot_configuration_space()
