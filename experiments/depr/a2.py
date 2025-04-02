import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
# print(plt.style.available)  # Lists available styles


class AttentionAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self._preprocess_data()
        self.output_dir = "rq1_plots"
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        self.colors = sns.color_palette("viridis", n_colors=6)

    # def _preprocess_data(self):
    #     """Clean and augment the tuning results data"""
    #     self.df['block_dims'] = self.df['block_size_x'].astype(str) + 'x' + self.df['block_size_y'].astype(str)
    #     self.df['threads_per_block'] = self.df['block_size_x'] * self.df['block_size_y']

    #     if 'seq_len' not in self.df.columns and 'n_steps' in self.df.columns:
    #         self.df['seq_len'] = self.df['n_steps']

        
    #     self.df.sort_values('threads_per_block', inplace=True)
    #     self.df['Joules_per_Occupancy'] = self.df['Joules/token'] / self.df['SM_Occupancy']

    def _preprocess_data(self):
        """Clean and augment the tuning results data"""
        # Create block dimensions string and thread count
        self.df['block_dims'] = self.df.apply(
            lambda row: f"{row['block_size_x']}x{row['block_size_y']}", axis=1
        )
        self.df['threads_per_block'] = self.df['block_size_x'] * self.df['block_size_y']

        # Sort by thread count then X dimension
        self.df.sort_values(['threads_per_block', 'block_size_x'], 
                        ascending=[True, True], inplace=True)
        
        # Generate ordered list of unique block dimensions
        self.block_dim_order = self.df['block_dims'].unique().tolist()
        
        # Calculate derived metrics
        if 'seq_len' not in self.df.columns:
            self.df['seq_len'] = self.df.get('n_steps', 512)  # Default seq_len
        self.df['Joules_per_Occupancy'] = self.df['Joules/token'] / self.df['SM_Occupancy']


    def plot_metric_vs_blocks(self, metric, ylabel):
        """Create individual line plots for each metric vs block dimensions"""
        fig, ax = plt.subplots()

        self.df['sort_key'] = self.df['block_dims'].apply(
        lambda x: (self.block_dim_order.index(x), x))
        
        # Plot lines with markers
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

        # X-axis formatting
        ax.set_xticks(range(len(self.block_dim_order)))
        ax.set_xticklabels(self.block_dim_order, rotation=45, ha='right')
        
        # Formatting
        ax.set_title(f'{ylabel} vs Block Dimensions')
        ax.set_xlabel('Block Dimensions (X x Y)')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.legend(title='Threads/Block', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        safe_metric = metric.replace('/', '_')
        fig.savefig(os.path.join(self.output_dir, f'block_{safe_metric}.png'))
        plt.close()

    def plot_energy_efficiency_tradeoff(self):
        """Enhanced 3D plot with better visualization"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sort data for better line visualization
        df_sorted = self.df.sort_values(['Joules/token', 'FLOPS/Watt'])
        
        # Create 3D scatter plot
        scatter = ax.scatter3D(
            df_sorted['Joules/token'],
            df_sorted['FLOPS/Watt'],
            df_sorted['SM_Occupancy'],
            c=df_sorted['threads_per_block'],
            cmap='viridis',
            s=50,
            edgecolor='k'
        )
        
        # Connect points with lines for better trend visibility
        ax.plot3D(
            df_sorted['Joules/token'],
            df_sorted['FLOPS/Watt'],
            df_sorted['SM_Occupancy'],
            'gray',
            alpha=0.3
        )

        # Axis labels and angles
        ax.set_xlabel('\nEnergy per Token (J)', linespacing=3)
        ax.set_ylabel('\nFLOPS/Watt', linespacing=3)
        ax.set_zlabel('\nSM Occupancy (%)', linespacing=3)
        ax.view_init(elev=25, azim=45)
        
        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Threads per Block', rotation=270, labelpad=15)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.output_dir, '3d_tradeoff.png'))
        plt.close()
        return fig

    def plot_time_vs_energy(self):
        """Enhanced time-energy plot with Pareto overlay"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get Pareto-optimal configurations
        pareto_mask = self._calculate_pareto_mask()
        pareto_df = self.df[pareto_mask]
        
        # Main scatter plot
        scatter = sns.scatterplot(
            data=self.df,
            x='time',
            y='nvml_energy',
            hue='block_dims',
            size='SM_Occupancy',
            palette='tab20',
            sizes=(40, 200),
            ax=ax
        )
        
        # Plot Pareto frontier
        sns.lineplot(
            data=pareto_df.sort_values('time'),
            x='time',
            y='nvml_energy',
            color='red',
            linestyle='--',
            label='Pareto Frontier',
            ax=ax
        )

        # Add annotations
        self._annotate_optimal_regions(ax)
        
        # Formatting
        ax.set_title('Time vs Energy Consumption by Block Shape')
        ax.set_xlabel('Execution Time (ms)')
        ax.set_ylabel('Energy Consumption (J)')
        ax.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_sm_vs_energy(self):
        """SM Occupancy vs Energy with architecture insights"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get Pareto points
        pareto_mask = self._calculate_pareto_mask()
        pareto_df = self.df[pareto_mask]
        
        # Create visualization
        scatter = sns.scatterplot(
            data=self.df,
            x='SM_Occupancy',
            y='Joules/token',
            hue='block_dims',
            size='FLOPS/Watt',
            palette='tab20',
            sizes=(20, 200),
            ax=ax
        )
        
        # Add Pareto frontier
        sns.lineplot(
            data=pareto_df.sort_values('SM_Occupancy'),
            x='SM_Occupancy',
            y='Joules/token',
            color='red',
            linestyle='--',
            label='Pareto Frontier',
            ax=ax
        )

        # Add annotations
        self._annotate_optimal_regions(ax)
        
        # Formatting
        ax.set_title('SM Occupancy vs Energy Efficiency')
        ax.set_xlabel('SM Occupancy (%)')
        ax.set_ylabel('Energy per Token (J)')
        ax.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def plot_block_metrics(self):
        """Create multi-panel plot of metrics vs block dimensions"""
        # Sort block dimensions by total threads
        # sorted_dims = self.df.sort_values('threads_per_block')['block_dims'].unique()
        
        # Create figure with 3x2 subplots
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
            ax.set_title(f'{label} vs Block Dimensions')
            ax.set_xlabel('Block Dimensions (X x Y)')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value annotations
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
        """Parallel coordinates plot for multi-metric analysis"""
        df = self.df.copy()
        
        # Normalize metrics for parallel coordinates
        metrics = ['time', 'nvml_energy', 'FLOPS/Watt', 'SM_Occupancy']
        df[metrics] = df[metrics].apply(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        # Sample for readability
        df = df.sort_values('threads_per_block').iloc[::3]
        
        fig = plt.figure(figsize=(12, 6))
        pd.plotting.parallel_coordinates(
            df[metrics + ['block_dims']],
            'block_dims',
            color=self.colors,
            linewidth=1.5
        )
        plt.title('Parallel Coordinates Analysis of Block Configurations')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        return fig

    def plot_pareto_frontier(self):
        """Enhanced Pareto frontier visualization with context length analysis"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by sequence length with log scale
        norm = plt.Normalize(np.log2(256), np.log2(4096))
        scatter = ax.scatter(
            self.df['Joules/token'], 
            self.df['FLOPS/Watt'],
            c=np.log2(self.df['seq_len']),
            cmap='viridis',
            alpha=0.7,
            s=100,
            edgecolor='w',
            linewidth=0.5
        )
        
        # Pareto filtering with epsilon-dominance
        pareto_mask = self._epsilon_dominated_filter()
        pareto_df = self.df[pareto_mask].sort_values('seq_len')
        
        # Plot Pareto frontier with sequence length grouping
        for seq_len, group in pareto_df.groupby('seq_len'):
            ax.plot(
                group['Joules/token'], 
                group['FLOPS/Watt'],
                linestyle='--',
                alpha=0.6,
                label=f'SeqLen {seq_len}'
            )

        # Formatting
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Log2(Sequence Length)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Energy per Token (Joules)')
        ax.set_ylabel('FLOPS/Watt')
        ax.legend(bbox_to_anchor=(1.05, 1))
        
        return fig

    def _epsilon_dominated_filter(self, epsilon=0.1):
        """Epsilon-dominance filter for practical Pareto frontier"""
        df = self.df.sort_values('Joules/token')
        mask = np.ones(len(df), dtype=bool)
        
        current_front = []
        for _, row in df.iterrows():
            dominated = False
            for cf in current_front:
                if (cf['Joules/token'] <= (1+epsilon)*row['Joules/token'] and 
                    cf['FLOPS/Watt'] >= (1-epsilon)*row['FLOPS/Watt']):
                    dominated = True
                    break
            if not dominated:
                current_front.append(row)
                mask[row.name] = True
            else:
                mask[row.name] = False
        return mask

    def plot_occupancy_energy_tradeoff(self):
        """2D projection of core RQ1 relationship"""
        fig, ax = plt.subplots(figsize=(10,6))
        
        # Color by sequence length, size by FLOPs/Watt
        scatter = sns.scatterplot(
            data=self.df,
            x='SM_Occupancy',
            y='Joules/token',
            hue='seq_len',
            size='FLOPS/Watt',
            palette='viridis',
            sizes=(20, 200),
            alpha=0.7
        )
        
        # Add Pareto frontier overlay
        pareto_df = self.df[self._epsilon_dominated_filter()]
        sns.lineplot(
            data=pareto_df.sort_values('SM_Occupancy'),
            x='SM_Occupancy',
            y='Joules/token',
            color='red',
            linestyle='--',
            label='Pareto Frontier'
        )
        
        # Annotate optimization regimes
        ax.annotate('High-Occupancy\nLow-Efficiency', 
                   xy=(0.8, 0.8), xycoords='axes fraction',
                   fontsize=10, color='darkred')
        ax.annotate('Optimal Balance Zone', 
                   xy=(0.4, 0.2), xycoords='axes fraction',
                   fontsize=12, color='darkgreen',
                   bbox=dict(boxstyle="round", fc="white", ec="green"))
        
        ax.set_xlabel('SM Occupancy (%)')
        ax.set_ylabel('Energy per Token (J)')
        ax.set_xscale('log')
        ax.set_title('RQ1 Core Tradeoff: SM Occupancy vs Energy Efficiency')
        plt.legend(bbox_to_anchor=(1.05, 1))
        return fig

    def _calculate_pareto_mask(self):
        """Identify non-dominated configurations"""
        objectives = self.df[['time', 'nvml_energy']]
        mask = np.ones(len(objectives), dtype=bool)
        
        for i, row in objectives.iterrows():
            if mask[i]:
                # Mark dominated points
                dominated = (
                    (objectives['time'] <= row['time']) & 
                    (objectives['nvml_energy'] <= row['nvml_energy'])
                )
                dominated[i] = False  # Exclude self
                mask[dominated] = False
        
        return mask

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

    def _perform_ancova(self):
        """Enhanced ANCOVA with occupancy interaction"""
        from statsmodels.formula.api import ols
        # Updated model with occupancy effects
        model = ols(
            'Q("Joules/token") ~ C(block_size_x) * SM_Occupancy + C(block_size_y) * SM_Occupancy + seq_len',
            data=self.df
        ).fit()
        return model.summary()

    def _bootstrap_confidence_intervals(self, n_boot=1000):
        """Bootstrap resampling for metric confidence intervals"""
        bootstraps = pd.DataFrame({
            'FLOPS/Watt': [self.df.sample(frac=1, replace=True)['FLOPS/Watt'].mean() 
                          for _ in range(n_boot)],
            'Joules/token': [self.df.sample(frac=1, replace=True)['Joules/token'].mean() 
                            for _ in range(n_boot)]
        })
        return bootstraps.quantile([0.025, 0.975])

    def plot_context_length_analysis(self):
        """Faceted analysis of sequence length impacts"""
        # Convert to categorical for discrete legend
        self.df['threads_cat'] = self.df['threads_per_block'].astype(str)
        
        g = sns.FacetGrid(
            self.df,
            col='seq_len',
            col_wrap=3,
            hue='threads_cat',
            palette='viridis',
            height=4,
            legend_out=False
        )
        
        # Plot without automatic legend
        g.map_dataframe(
            sns.scatterplot, 
            x='SM_Occupancy', 
            y='Joules/token', 
            size='FLOPS/Watt',
            alpha=0.6,
            legend=False
        )
        
        # Add custom legends
        handles, labels = zip(*[
            (plt.Line2D([], [], marker='o', color='k', markersize=np.sqrt(s/1e3), linestyle=''),
            f"{s/1e3:.1f}K FLOPS/W")
            for s in self.df['FLOPS/Watt'].quantile([0.25, 0.5, 0.75])
        ])
        
        g.fig.legend(
            handles, labels,
            title='FLOPS/Watt',
            bbox_to_anchor=(1.05, 0.5),
            loc='center left'
        )
        
        # Add thread count legend
        g.add_legend(
            title="Threads/Block",
            label_order=sorted(self.df['threads_cat'].unique()),
            bbox_to_anchor=(1.05, 0.8)
        )
        
        g.set_titles("Sequence Length: {col_name}")
        g.set_axis_labels("SM Occupancy (%)", "Energy per Token (J)")
        
        return g.fig

    def save_plots(self):
        """Generate and save all plots in separate files"""
        # Metric vs block dimensions series
        metrics = [
            ('time', 'Execution Time (ms)'),
            ('nvml_energy', 'Energy Consumption (J)'),
            ('Joules/token', 'Energy per Token (J)'),
            ('FLOPS/Watt', 'FLOPS per Watt'),
            ('SM_Occupancy', 'SM Occupancy (%)'),
            ('Joules_per_Occupancy', 'Joules per Occupancy Unit')
        ]
        
        # Generate basic metric plots
        for metric, label in metrics:
            self.plot_metric_vs_blocks(metric, label)

        # Complex analysis plots
        plot_functions = [
            self.plot_energy_efficiency_tradeoff,
            self.plot_time_vs_energy,
            self.plot_sm_vs_energy,
            self.plot_occupancy_energy_tradeoff,
            # self.plot_context_length_analysis,
            self.plot_pareto_frontier,
            self.plot_block_metrics,
            self.plot_parallel_coordinates
        ]

        # Generate and save complex plots
        for plot_func in plot_functions:
            print(f"Generating {plot_func.__name__} plot...")
            fig = plot_func()
            fig.savefig(os.path.join(
                self.output_dir,
                f"{plot_func.__name__[5:]}.png"  # Remove "plot_" prefix
            ))
            plt.close(fig)

if __name__ == "__main__":
    # Load and analyze tuning results
    analyzer = AttentionAnalyzer("rq1_data/mha_tuning_results.csv")
    
    # Generate visualizations
    analyzer.save_plots()
    
    # Generate statistical report
    report = analyzer.generate_analysis_report()
    
    # Save optimal configurations
    report['pareto_configs'].to_csv("rq1_data/rq1_optimal_configs.csv", index=False)
    
    # Print key insights
    print("RQ1 Analysis Complete. Key Findings:")
    print(f"Identified {len(report['pareto_configs'])} Pareto-optimal configurations")
    print("ANCOVA Results:")
    print(report['ancova_results'].tables[1])
    print("\n95% Confidence Intervals:")
    print(report['bootstrap_ci'])