import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import json
# print(plt.style.available)  # Lists available styles


class AttentionAnalyzer:
    def __init__(self, json_path):
        # Keep all preprocessing from new script
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        rows = []
        for key, value in data['cache'].items():
            params = key.split(',')
            row = {
                'block_size_x': int(params[0]),
                'block_size_y': int(params[1]),
                'nvml_pwr_limit': int(params[2]),
            }
            row.update(value)
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        self._preprocess_data()
        self.output_dir = "/home/anonymous/flipflop/cuda_kernel_energy_empirical/rq1_plots/multihead_attention"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Adopt old script's visual style parameters
        plt.style.use('seaborn-v0_8-paper')
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
        self.colors = sns.color_palette("viridis")

    def _preprocess_data(self):
        """Clean and augment the tuning results data"""
        # Create block dimensions string and thread count
        self.df['block_dims'] = self.df.apply(
            lambda row: f"{row['block_size_x']}x{row['block_size_y']}", axis=1
        )
        self.df['threads_per_block'] = self.df['block_size_x'] * self.df['block_size_y']
        self.df['nvml_pwr_limit'] = self.df['nvml_pwr_limit'].astype(int)  # Ensure integer type

        # Sort by thread count then X dimension
        self.df.sort_values(['threads_per_block', 'block_size_x'], 
                        ascending=[True, True], inplace=True)
        
        # Generate ordered list of unique block dimensions
        self.block_dim_order = self.df['block_dims'].unique().tolist()
        
        # Calculate derived metrics (handle NaNs)
        self.df['SM_Occupancy'] = self.df['sm__warps_active.avg.pct_of_peak_sustained_active'].fillna(0)
        if 'seq_len' not in self.df.columns:
            self.df['seq_len'] = self.df.get('n_steps', 512)  # Default seq_len
        self.df['Joules_per_Occupancy'] = self.df['nvml_energy'] / self.df['SM_Occupancy'].replace(0, np.nan)
        self.df['Joules_per_Occupancy'] = self.df['Joules_per_Occupancy'].fillna(0)


    def plot_metric_vs_blocks(self, metric, ylabel):
        fig, ax = plt.subplots()

        self.df['sort_key'] = self.df['block_dims'].apply(
            lambda x: (self.block_dim_order.index(x), x))
        
        # Use lineplot with threads_per_block hue 
        sns.lineplot(
            data=self.df.sort_values('sort_key'), 
            x='block_dims',
            y=metric,
            hue='threads_per_block',
            palette=self.colors,
            marker='o',
            markersize=8,
            ax=ax
        )

        ax.set_xticks(range(len(self.block_dim_order)))
        ax.set_xticklabels(self.block_dim_order, rotation=45, ha='right')
        ax.set_title(f'{ylabel} vs Block Dimensions')
        ax.legend(title='Threads/Block', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        # metric replace "/" with "_" for filename
        metric = metric.replace("/", "_")
        fig.savefig(os.path.join(self.output_dir, f'block_{metric}.png'))
        plt.close()

    def plot_energy_efficiency_tradeoff(self):
        """3D plot with power limit as color"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        df_sorted = self.df.sort_values(['Joules/token', 'FLOPS/Watt'])
        
        # Create 3D scatter plot with power limit as color
        scatter = ax.scatter3D(
            df_sorted['Joules/token'],
            df_sorted['FLOPS/Watt'],
            df_sorted['SM_Occupancy'],
            c=df_sorted['nvml_pwr_limit'],
            cmap='viridis',
            s=50,
            edgecolor='k'
        )
        
        # Axis labels
        ax.set_xlabel('\nEnergy per Token (J)', linespacing=3)
        ax.set_ylabel('\nFLOPS/Watt', linespacing=3)
        ax.set_zlabel('\nSM Occupancy (%)', linespacing=3)
        ax.view_init(elev=25, azim=45)
        
        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Power Limit (W)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, '3d_tradeoff.png'))
        plt.close()
        return fig

    def _annotate_optimal_regions(self, ax):
        """Enhanced annotations with block shape examples"""
        # Find optimal points
        min_energy = self.df['Joules/token'].min()
        max_flops = self.df['FLOPS/Watt'].max()
        
        # Energy-optimal annotation
        energy_optimal = self.df[self.df['Joules/token'] == min_energy].iloc[0]
        ax.annotate(
            f"Most Efficient\n{energy_optimal['block_dims']}",
            (energy_optimal['SM_Occupancy'], energy_optimal['Joules/token']),
            xytext=(10, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->")
        )
        
        # Performance-optimal annotation
        flops_optimal = self.df[self.df['FLOPS/Watt'] == max_flops].iloc[0]
        ax.annotate(
            f"Highest FLOPS/W\n{flops_optimal['block_dims']}",
            (flops_optimal['SM_Occupancy'], flops_optimal['Joules/token']),
            xytext=(-80, 20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->")
        )

    def _epsilon_dominated_filter(self, epsilon=0.1):
        """Epsilon-dominance filter for practical Pareto frontier"""
        df = self.df.sort_values('nvml_energy')
        mask = np.ones(len(df), dtype=bool)
        
        current_front = []
        for _, row in df.iterrows():
            dominated = False
            for cf in current_front:
                if (cf['nvml_energy'] <= (1+epsilon)*row['nvml_energy'] and 
                    cf['flops_watt'] >= (1-epsilon)*row['flops_watt']):
                    dominated = True
                    break
            if not dominated:
                current_front.append(row)
                mask[row.name] = True
            else:
                mask[row.name] = False
        return mask
    
    def generate_analysis_report(self):
        """Create comprehensive statistical analysis report"""
        report = {
            'pareto_configs': self._get_pareto_configs(),
            'ancova_results': self._perform_ancova(),
            'bootstrap_ci': self._bootstrap_confidence_intervals()
        }
        return report
    
    def _get_pareto_configs(self):
        """Identify optimal configurations meeting RQ1 criteria"""
        return self.df[
            (self.df['FLOPS/Watt'] >= 0.9*self.df['FLOPS/Watt'].max()) &
            (self.df['Joules/token'] <= 1.1*self.df['Joules/token'].min())
        ][['block_size_x', 'block_size_y', 'SM_Occupancy', 'Joules/token', 'FLOPS/Watt']]

    def _bootstrap_confidence_intervals(self, n_boot=1000):
        """Bootstrap resampling for metric confidence intervals"""
        bootstraps = pd.DataFrame({
            'FLOPS/Watt': [self.df.sample(frac=1, replace=True)['FLOPS/Watt'].mean() 
                          for _ in range(n_boot)],
            'Joules/token': [self.df.sample(frac=1, replace=True)['Joules/token'].mean() 
                            for _ in range(n_boot)]
        })
        return bootstraps.quantile([0.025, 0.975])


    def plot_time_vs_energy(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.scatterplot(
            data=self.df,
            x='time',
            y='nvml_energy',
            hue='block_dims',
            size='SM_Occupancy',
            palette='tab20',
            sizes=(40, 200),
            ax=ax
        )
        
        pareto_mask = self._calculate_pareto_mask()
        pareto_df = self.df[pareto_mask]
        sns.lineplot(
            data=pareto_df.sort_values('time'),
            x='time',
            y='nvml_energy',
            color='red',
            linestyle='--',
            label='Pareto Frontier',
            ax=ax
        )

        ax.set_title('Time vs Energy Consumption by Block Shape')
        ax.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_sm_vs_energy(self):
        """Old color scheme with new data relationships"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.scatterplot(
            data=self.df,
            x='SM_Occupancy',
            y='Joules/token',
            hue='block_dims',
            size='FLOPS/Watt',
            palette='tab20',
            sizes=(20, 200),
            ax=ax
        )
        
        # Keep Pareto frontier from new implementation
        pareto_df = self.df[self._calculate_pareto_mask()]
        sns.lineplot(
            data=pareto_df.sort_values('SM_Occupancy'),
            x='SM_Occupancy',
            y='Joules/token',
            color='red',
            linestyle='--',
            label='Pareto Frontier',
            ax=ax
        )

        ax.set_title('SM Occupancy vs Energy Efficiency')
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_block_metrics(self):
        """Revert to barplot visualization but keep power data"""
        fig, axs = plt.subplots(3, 2, figsize=(12, 14))
        metrics = [
            ('time', 'Execution Time (ms)'),
            ('nvml_energy', 'Energy (J)'), 
            ('nvml_power', 'Power (W)'),
            ('Joules/token', 'Energy per Token (J)'),
            ('FLOPS/Watt', 'FLOPS/Watt'),
            ('SM_Occupancy', 'SM Occupancy (%)')
        ]
        
        for idx, (metric, label) in enumerate(metrics):
            ax = axs[idx//2, idx%2]
            sns.barplot(
                data=self.df,
                x='block_dims',
                y=metric,
                order=self.block_dim_order,
                palette=self.colors,
                ax=ax
            )
            # Keep value annotations from new script
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom',
                    fontsize=8
                )
        plt.tight_layout()
        return fig

    def plot_parallel_coordinates(self):
        """Parallel coordinates plot with power limit integration"""
        df = self.df.copy()
        
        # Normalize metrics and include power limit
        metrics = ['time', 'nvml_energy', 'FLOPS/Watt', 'SM_Occupancy', 'nvml_pwr_limit']
        df[metrics] = df[metrics].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        
        # Use power limit for coloring
        pd.plotting.parallel_coordinates(
            df[metrics + ['block_dims']],
            'block_dims',
            color=self.colors,
            ax=ax,
            linewidth=1.5
        )
        
        plt.title('Parallel Coordinates Analysis (Color: Power Limit)')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_pareto_frontier(self):
        """Pareto frontier visualization with power limit coloring"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by power limit
        scatter = ax.scatter(
            self.df['Joules/token'], 
            self.df['FLOPS/Watt'],
            c=self.df['nvml_pwr_limit'],
            cmap='viridis',
            alpha=0.7,
            s=100,
            edgecolor='w',
            linewidth=0.5
        )
        
        # Pareto filtering
        pareto_mask = self._epsilon_dominated_filter()
        pareto_df = self.df[pareto_mask]
        
        # Plot Pareto frontier with power limit
        sns.lineplot(
            data=pareto_df.sort_values('Joules/token'),
            x='Joules/token',
            y='FLOPS/Watt',
            hue='nvml_pwr_limit',
            palette='viridis',
            legend=False,
            ax=ax
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Power Limit (W)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy per Token (Joules)')
        ax.set_ylabel('FLOPS/Watt')
        ax.set_title('Pareto Frontier with Power Limit Coloring')
        return fig

    def plot_occupancy_energy_tradeoff(self):
        """SM Occupancy vs Energy with power limit integration"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color by power limit
        scatter = sns.scatterplot(
            data=self.df,
            x='SM_Occupancy',
            y='Joules/token',
            hue='nvml_pwr_limit',
            size='FLOPS/Watt',
            palette='viridis',
            sizes=(20, 200),
            ax=ax
        )
        
        # Pareto frontier
        pareto_df = self.df[self._epsilon_dominated_filter()]
        sns.lineplot(
            data=pareto_df.sort_values('SM_Occupancy'),
            x='SM_Occupancy',
            y='Joules/token',
            color='red',
            linestyle='--',
            label='Pareto Frontier',
            ax=ax
        )

        ax.set_xlabel('SM Occupancy (%)')
        ax.set_ylabel('Energy per Token (J)')
        ax.set_title('Occupancy vs Energy Efficiency (Color: Power Limit)')
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_combined_utilization_energy(self):
        """Combined SM utilization and energy per token plot"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Sort data using precomputed block order
        plot_df = self.df.set_index('block_dims').loc[self.block_dim_order].reset_index()
        
        # SM Utilization (Bar plot)
        ax1.bar(
            plot_df['block_dims'],
            plot_df['SM_Occupancy'],
            color='#1f77b4',  # Matplotlib blue
            alpha=0.7,
            label='SM Occupancy'
        )
        ax1.set_ylabel('SM Occupancy (%)', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.set_ylim(0, 100)

        # Energy per Token (Line plot)
        ax2 = ax1.twinx()
        ax2.plot(
            plot_df['block_dims'],
            plot_df['Joules/token'],
            color='#d62728',  # Matplotlib red
            marker='o',
            markersize=8,
            linewidth=2,
            label='Energy/Token'
        )
        ax2.set_ylabel('Energy per Token (J)', color='#d62728') 
        ax2.tick_params(axis='y', labelcolor='#d62728')

        # Formatting
        ax1.set_xlabel('Block Dimensions (X x Y)', fontsize=12)
        ax1.set_xticklabels(self.block_dim_order, rotation=45, ha='right')
        ax1.grid(True, axis='y', alpha=0.3)
        plt.title('SM Utilization and Energy Efficiency by Block Configuration', pad=20)
        
        # Combined legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0.01, 0.99))

        # Highlight optimal region
        min_energy_idx = plot_df['Joules/token'].idxmin()
        ax2.annotate('Most Efficient', 
                    (plot_df['block_dims'][min_energy_idx], plot_df['Joules/token'][min_energy_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", color='#2ca02c'))

        plt.tight_layout()
        return fig


    def _calculate_pareto_mask(self):
        """Updated Pareto mask considering power limits"""
        # Create composite objective of energy efficiency and power usage
        self.df['composite_score'] = (
            self.df['FLOPS/Watt'] / self.df['nvml_pwr_limit']
        )
        
        # Find non-dominated points considering time, energy, and composite score
        objectives = self.df[['time', 'nvml_energy', 'composite_score']]
        mask = np.ones(len(objectives), dtype=bool)
        
        for i, row in objectives.iterrows():
            if mask[i]:
                dominated = (
                    (objectives['time'] <= row['time']) & 
                    (objectives['nvml_energy'] <= row['nvml_energy']) &
                    (objectives['composite_score'] >= row['composite_score'])
                )
                dominated[i] = False
                mask[dominated] = False
        
        return mask

    def _perform_ancova(self):
        """Updated ANCOVA with power limit effects"""
        from statsmodels.formula.api import ols
        model = ols(
            'Q("Joules/token") ~ C(block_size_x) * SM_Occupancy + '
            'C(block_size_y) * SM_Occupancy + nvml_pwr_limit',
            data=self.df
        ).fit()
        return model.summary()

    def plot_context_length_analysis(self):
        """Faceted analysis by power limit instead of sequence length"""
        g = sns.FacetGrid(
            self.df,
            col='nvml_pwr_limit',
            col_wrap=4,
            hue='block_dims',
            palette='viridis',
            height=5,
            aspect=0.8
        )
        
        g.map_dataframe(
            sns.scatterplot, 
            x='SM_Occupancy', 
            y='Joules/token', 
            size='FLOPS/Watt',
            alpha=0.7
        )
        
        g.set_titles("Power Limit: {col_name}W")
        g.set_axis_labels("SM Occupancy (%)", "Energy per Token (J)")
        g.add_legend(title="Block Dimensions", bbox_to_anchor=(1.05, 0.5))
        
        return g.fig

    def save_plots(self):
        """Generate and save all plots"""
        metrics = [
            ('time', 'Execution Time (ms)'),
            ('nvml_energy', 'Energy Consumption (J)'),
            ('Joules/token', 'Energy per Token (J)'),
            ('FLOPS/Watt', 'FLOPS per Watt'),
            ('SM_Occupancy', 'SM Occupancy (%)'),
            ('Joules_per_Occupancy', 'Joules per Occupancy Unit')
        ]
        
        for metric, label in metrics:
            self.plot_metric_vs_blocks(metric, label)

        plot_functions = [
            self.plot_energy_efficiency_tradeoff,
            self.plot_time_vs_energy,
            self.plot_sm_vs_energy,
            self.plot_occupancy_energy_tradeoff,
            self.plot_pareto_frontier,
            self.plot_block_metrics,
            self.plot_parallel_coordinates
        ]

        for plot_func in plot_functions:
            fig = plot_func()
            fig.savefig(os.path.join(
                self.output_dir,
                f"{plot_func.__name__[5:]}.png"
            ))
            plt.close(fig)

        fig = self.plot_combined_utilization_energy()
        fig.savefig(os.path.join(self.output_dir, 'combined_utilization_energy.png'))
        plt.close(fig)


class AttentionAnalyzerEnhanced(AttentionAnalyzer):
    def plot_dynamic_vs_static_efficiency(self):
        """Compare adaptive configurations against common static heuristics"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Identify common static configurations
        static_blocks = ['32x32', '16x16', '64x4', '128x2']
        static_mask = self.df['block_dims'].isin(static_blocks)
        adaptive_mask = self._epsilon_dominated_filter()
        
        # Plot efficiency comparison
        sns.scatterplot(
            data=self.df[static_mask],
            x='SM_Occupancy',
            y='nvml_energy',
            hue='block_dims',
            style='block_dims',
            s=200,
            ax=ax,
            legend='full'
        )
        
        sns.scatterplot(
            data=self.df[adaptive_mask],
            x='SM_Occupancy',
            y='nvml_energy',
            color='purple',
            s=100,
            edgecolor='k',
            label='Adaptive Frontier',
            ax=ax
        )
        
        ax.set_title('Static vs Adaptive Configuration Efficiency')
        ax.annotate('Adaptive Advantage Zone', (55, 0.8*self.df['nvml_energy'].min()),
                   xytext=(40, 1.2*self.df['nvml_energy'].min()),
                   arrowprops=dict(arrowstyle="->"), fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_power_adaptive_frontiers(self):
        """Show how optimal configurations shift with power limits"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by power limit and find optimal configs
        power_levels = sorted(self.df['nvml_pwr_limit'].unique())
        palette = sns.color_palette("rocket_r", len(power_levels))
        
        for pwr, color in zip(power_levels, palette):
            subset = self.df[self.df['nvml_pwr_limit'] == pwr]
            pareto_mask = self._epsilon_dominated_filter_subset(subset)
            sns.lineplot(
                data=subset[pareto_mask].sort_values('Joules/token'),
                x='SM_Occupancy',
                y='Joules/token',
                color=color,
                label=f'{pwr}W',
                ax=ax
            )
        
        ax.set_title('Power-Adaptive Pareto Frontier Shifts')
        ax.set_xlabel('SM Occupancy (%)')
        ax.set_ylabel('Energy per Token (J)')
        plt.legend(title='Power Limit', bbox_to_anchor=(1.05, 1))
        return fig

    def _epsilon_dominated_filter_subset(self, subset, epsilon=0.1):
        """Apply epsilon-dominance filtering to a subset"""
        df = subset.sort_values('nvml_energy')
        mask = np.ones(len(df), dtype=bool)
        current_front = []
        
        for _, row in df.iterrows():
            dominated = False
            for cf in current_front:
                if (cf['nvml_energy'] <= (1+epsilon)*row['nvml_energy'] and 
                    cf['flops_watt'] >= (1-epsilon)*row['flops_watt']):
                    dominated = True
                    break
            if not dominated:
                current_front.append(row)
                mask[row.name] = True
            else:
                mask[row.name] = False
        return mask

    def plot_adaptation_trajectory(self):
        """Visualize configuration changes across power limits"""
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Main trajectory plot
        sns.lineplot(
            data=self.df,
            x='nvml_pwr_limit',
            y='Joules/token',
            hue='block_dims',
            estimator='min',
            err_style='band',
            ax=ax1
        )
        ax1.set_title('Energy Efficiency Adaptation Trajectory')
        
        # Block size distribution
        sns.histplot(
            self.df.groupby('nvml_pwr_limit')['block_dims'].agg(pd.Series.mode),
            ax=ax2,
            discrete=True
        )
        ax2.set_title('Most Frequent Optimal Blocks')
        
        # SM occupancy vs power
        sns.boxplot(
            data=self.df,
            x='nvml_pwr_limit',
            y='SM_Occupancy',
            ax=ax3,
            palette='viridis'
        )
        ax3.set_title('Occupancy Distribution by Power Limit')
        
        plt.tight_layout()
        return fig

    def plot_efficiency_gains(self):
        """Quantify improvements from adaptive configurations"""
        static_blocks = ['32x32', '16x16', '64x4']
        static_df = self.df[self.df['block_dims'].isin(static_blocks)]
        adaptive_df = self.df[self._epsilon_dominated_filter()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate improvement percentages
        gains = []
        for metric in ['Joules/token', 'FLOPS/Watt', 'SM_Occupancy']:
            static_mean = static_df[metric].mean()
            adaptive_mean = adaptive_df[metric].mean()
            improvement = 100*(static_mean - adaptive_mean)/static_mean
            gains.append(improvement)
        
        sns.barplot(
            x=['Energy/Token', 'FLOPS/Watt', 'SM Occupancy'],
            y=gains,
            palette=['#1f77b4', '#ff7f0e', '#2ca02c'],
            ax=ax
        )
        
        ax.set_ylabel('% Improvement from Adaptation')
        ax.set_title('Performance Gains of Adaptive Configurations')
        plt.xticks(rotation=45)
        return fig

    def save_enhanced_plots(self):
        """Save all enhanced plots"""
        new_plots = [
            self.plot_dynamic_vs_static_efficiency,
            self.plot_power_adaptive_frontiers,
            self.plot_adaptation_trajectory,
            self.plot_efficiency_gains
        ]
        
        for plot_func in new_plots:
            fig = plot_func()
            fig.savefig(os.path.join(
                self.output_dir,
                f"enhanced_{plot_func.__name__[5:]}.png"
            ))
            plt.close(fig)
        super().save_plots()


if __name__ == "__main__":
    analyzer = AttentionAnalyzerEnhanced(
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/power_tuning_float16.json"
    )
    analyzer.save_enhanced_plots()


# if __name__ == "__main__":
#     analyzer = AttentionAnalyzer("/home/anonymous/flipflop/cuda_kernel_energy_empirical/rq1_data/multihead_attention/mha_cache.json")  # Changed to JSON
#     analyzer.save_plots()
    
#     report = analyzer.generate_analysis_report()
#     report['pareto_configs'].to_csv("/home/anonymous/flipflop/cuda_kernel_energy_empirical/rq1_data/multihead_attention/rq1_optimal_configs.csv", index=False)
    
#     print("RQ1 Analysis Complete. Key Findings:")
#     print(f"Identified {len(report['pareto_configs'])} Pareto-optimal configurations")
#     print("ANCOVA Results:")
#     print(report['ancova_results'].tables[1])
#     print("\n95% Confidence Intervals:")
#     print(report['bootstrap_ci'])


