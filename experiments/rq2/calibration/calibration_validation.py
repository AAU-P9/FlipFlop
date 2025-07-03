import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os

def load_calibration_data(pattern="calibration_pl*.json"):
    """Load all calibration files into a structured DataFrame"""
    records = []
    for fpath in glob.glob(pattern):
        print("*" * 20)
        print(f"Loading calibration data from {fpath}")
        with open(fpath) as f:
            data = json.load(f)
        for arch in data.values():
            arch['power_limit'] = int(fpath.split('pl')[-1].split('.')[0])
            records.append(arch)
    return pd.DataFrame(records)

def analyze_power_limit_relationships(df):
    """Statistical analysis of parameter trends across power limits"""
    analysis = {}
    
    # Parameter Stability Analysis
    stable_params = ['Mem_LD_coal_ns', 'Mem_LD_uncoal_ns', 'issue_cycles']
    for param in stable_params:
        slope, intercept, r_value, p_value, _ = stats.linregress(
            df['power_limit'], df[param]
        )
        analysis[param] = {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'mean': df[param].mean(),
            'std': df[param].std()
        }
    
    return analysis

def plot_cross_pl_trends(df, analysis_results, out_dir="./"):
    """Generate robust visualization suite"""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Parameter Stability Plot
    plt.figure(figsize=(12, 6))
    for param in analysis_results:
        plt.plot(df['power_limit'], df[param], 'o-', label=param, markersize=8)
    plt.title("Architectural Parameter Stability Across Power Limits\n"
             "(Coalesced/Uncoalesced Memory Latencies and Issue Cycles)")
    plt.xlabel("Power Limit (W)", fontsize=12)
    plt.ylabel("Parameter Value (ns/cycles)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.savefig(f"{out_dir}/parameter_stability.png", bbox_inches='tight')
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 6))
    corr_matrix = df[['Mem_LD_coal_ns', 'Mem_LD_uncoal_ns',
                     'effective_mem_bw_gbps', 'power_limit']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                fmt=".2f", vmin=-1, vmax=1)
    plt.title("Memory Characteristics Correlation Matrix")
    plt.savefig(f"{out_dir}/memory_correlation_heatmap.png")
    plt.close()

    # 3. Scatter Matrix with Regression
    g = sns.PairGrid(df, vars=['Mem_LD_coal_ns', 'Mem_LD_uncoal_ns',
                              'effective_mem_bw_gbps'], 
                    hue='power_limit', palette='viridis')
    g.map_upper(sns.scatterplot, s=50, alpha=0.8)
    g.map_lower(sns.regplot, scatter_kws={'s': 40, 'alpha': 0.6},
                line_kws={'color': 'red'})
    g.map_diag(sns.histplot, kde=False, bins=6)
    g.add_legend()
    g.fig.suptitle("Memory Characteristics Relationships", y=1.02)
    g.savefig(f"{out_dir}/memory_characteristics_scattermatrix.png")
    plt.close()

def plot_power_law( xs, ys, alpha, beta):
        """Generate and save a plot of SM concurrency power law validation."""
        
        if len(xs) == 0 or len(ys) == 0:
            print("[WARNING] No data to plot for power law validation.")
            return

        plt.figure(figsize=(10, 6))
        # Plot measured data points
        plt.scatter(xs, ys, color='red', label='Measured Power Delta', s=200, alpha=0.7, edgecolor='black')
        # Generate fitted curve
        x_fit = np.linspace(min(xs), max(xs), 100)
        # change size of x axis numbers
        plt.xticks(fontsize=20)

        y_fit = alpha * np.power(x_fit, beta)
        plt.yticks(fontsize=20)
        plt.plot(x_fit, y_fit, 'b-', label=f'Fitted Curve: {alpha:.2f} * x^{beta:.2f}', linewidth=5)

        # chanhe legend label size

        
        plt.xlabel('Number of Active SMs', fontsize=24)
        plt.ylabel('Power Delta (W)', fontsize=24)
        # plt.title(f'Power Law Validation', fontsize=18)
        plt.legend(fontsize=22, loc='lower right')
        plt.grid(True)

        # Save the plot next to the calibration file
        # base_name = os.path.splitext(self.calibration_file)[0]
        plot_path = f"scale_sm_power_law.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved power law validation plot to {plot_path}")

def main():
    # Load and analyze data
    cal_df = load_calibration_data()
    print("Loaded DataFrame columns:", cal_df.columns.tolist())
    # print cal_df keys
    # plot_power_law(cal_df["sm_values"], cal_df["power_deltas"], cal_df["power_alpha"], cal_df["power_beta"])

    for _, row in cal_df.iterrows():
        plot_power_law(
            xs=row['sm_values'],
            ys=row['power_deltas'],
            alpha=row['power_alpha'],
            beta=row['power_beta']
        )

    # analysis = analyze_power_limit_relationships(cal_df)
    
    # # Generate visualizations
    # plot_cross_pl_trends(cal_df, analysis)
    
    # # Save and print statistics
    # stats_df = pd.DataFrame(analysis).T
    # stats_df.to_csv("parameter_stability_stats.csv")
    
    # print("\nAnalysis Completed Successfully")
    # print(f"Coalesced Latency R²: {stats_df.loc['Mem_LD_coal_ns', 'r_squared']:.3f}")
    # print(f"Memory Bandwidth-Power Correlation: {cal_df[['effective_mem_bw_gbps','power_limit']].corr().iloc[0,1]:.2f}")

if __name__ == "__main__":
    main()