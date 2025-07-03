import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_rq3_analysis(csv_path, out_dir="rq3_plots"):
    """Generate comparative plots for RQ3 energy model validation"""
    # Load and prepare data
    df = pd.read_csv(csv_path)
    valid_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Filter valid block configurations
    df = df[
        df["block_x"].isin(valid_blocks) & 
        df["block_y"].isin(valid_blocks)
    ].copy()
    
    # Create composite fields
    df["block_dims"] = df["block_x"].astype(str) + "x" + df["block_y"].astype(str)
    df["threads_per_block"] = df["block_x"] * df["block_y"]
    df = df.sort_values(["threads_per_block", "block_x"])
    df["block_dims"] = pd.Categorical(df["block_dims"], ordered=True, 
                                    categories=df["block_dims"].unique())

    # Calculate metrics
    df["predicted_energy"] = (df["predicted_power"] * df["predicted_time_ns"]/1e9) / (df["batch_size"] * df["seq_len"])
    df["actual_energy"] = (df["actual_power"] * df["actual_time_ns"]/1e9) / (df["batch_size"] * df["seq_len"])
    df["predicted_time_s"] = df["predicted_time_ns"]/1e9
    df["actual_time_s"] = df["actual_time_ns"]/1e9

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Generate plots for each metric
    metrics = [
        ("energy", "Energy per Token (Joules/token)", "predicted_energy", "actual_energy"),
        ("power", "Power Consumption (Watts)", "predicted_power", "actual_power"),
        ("time", "Execution Time (seconds)", "predicted_time_s", "actual_time_s")
    ]

    for metric, ylabel, pred_col, actual_col in metrics:
        # Combined plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df.melt(
                id_vars=["block_dims", "seq_len"],
                value_vars=[pred_col, actual_col],
                var_name="type",
                value_name=metric
            ),
            x="block_dims",
            y=metric,
            hue="seq_len",
            style="type",
            markers=True,
            palette="tab10",
            estimator=None
        )
        plt.title(f"{ylabel} - Predicted vs Actual")
        plt.xticks(rotation=45)
        plt.xlabel("Block Dimensions")
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/combined_{metric}.png", dpi=150)
        plt.close()

        # Individual predicted plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="block_dims",
            y=pred_col,
            hue="seq_len",
            marker="o",
            palette="tab10"
        )
        plt.title(f"Predicted {ylabel}")
        plt.xticks(rotation=45)
        plt.xlabel("Block Dimensions")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/predicted_{metric}.png", dpi=150)
        plt.close()

        # Individual actual plot  
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="block_dims",
            y=actual_col,
            hue="seq_len",
            marker="o",
            palette="tab10"
        )
        plt.title(f"Actual {ylabel}")
        plt.xticks(rotation=45)
        plt.xlabel("Block Dimensions")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/actual_{metric}.png", dpi=150)
        plt.close()


def plot_rq3_analysis_normalized(csv_path, out_dir="rq3_plots_norm"):
    """Generate comparative normalized plots for RQ3 energy model validation.
    For each sequence length, both predicted and actual values are normalized (min-max)
    so that trends can be directly compared on the same scale.
    """
    # Load and prepare data
    df = pd.read_csv(csv_path)
    valid_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_x"].isin(valid_blocks) &
        df["block_y"].isin(valid_blocks)
    ].copy()
    
    # Create composite fields for plotting
    df["block_dims"] = df["block_x"].astype(str) + "x" + df["block_y"].astype(str)
    df["threads_per_block"] = df["block_x"] * df["block_y"]
    df = df.sort_values(["threads_per_block", "block_x"])
    df["block_dims"] = pd.Categorical(df["block_dims"], ordered=True, 
                                      categories=df["block_dims"].unique())
    
    # Calculate raw metrics (e.g., energy, power, execution time in seconds)
    df["predicted_energy"] = (df["predicted_power"] * df["predicted_time_ns"] / 1e9) / (df["batch_size"] * df["seq_len"])
    df["actual_energy"] = (df["actual_power"] * df["actual_time_ns"] / 1e9) / (df["batch_size"] * df["seq_len"])
    df["predicted_time_s"] = df["predicted_time_ns"] / 1e9
    df["actual_time_s"] = df["actual_time_ns"] / 1e9

    # Create normalized metrics for each seq_len group using min-max scaling.
    # This procedure preserves trend differences while mapping values to [0,1].
    for seq in df["seq_len"].unique():
        mask = df["seq_len"] == seq
        # Normalize energy
        actual_min = df.loc[mask, "actual_energy"].min()
        actual_max = df.loc[mask, "actual_energy"].max()
        df.loc[mask, "actual_energy_norm"] = (df.loc[mask, "actual_energy"] - actual_min) / (actual_max - actual_min + 1e-9)
        
        pred_min = df.loc[mask, "predicted_energy"].min()
        pred_max = df.loc[mask, "predicted_energy"].max()
        df.loc[mask, "predicted_energy_norm"] = (df.loc[mask, "predicted_energy"] - pred_min) / (pred_max - pred_min + 1e-9)
        
        # Similarly, you can normalize power and time if desired:
        actual_p_min = df.loc[mask, "actual_power"].min()
        actual_p_max = df.loc[mask, "actual_power"].max()
        df.loc[mask, "actual_power_norm"] = (df.loc[mask, "actual_power"] - actual_p_min) / (actual_p_max - actual_p_min + 1e-9)
        
        pred_p_min = df.loc[mask, "predicted_power"].min()
        pred_p_max = df.loc[mask, "predicted_power"].max()
        df.loc[mask, "predicted_power_norm"] = (df.loc[mask, "predicted_power"] - pred_p_min) / (pred_p_max - pred_p_min + 1e-9)
        
        actual_t_min = df.loc[mask, "actual_time_s"].min()
        actual_t_max = df.loc[mask, "actual_time_s"].max()
        df.loc[mask, "actual_time_norm"] = (df.loc[mask, "actual_time_s"] - actual_t_min) / (actual_t_max - actual_t_min + 1e-9)
        
        pred_t_min = df.loc[mask, "predicted_time_s"].min()
        pred_t_max = df.loc[mask, "predicted_time_s"].max()
        df.loc[mask, "predicted_time_norm"] = (df.loc[mask, "predicted_time_s"] - pred_t_min) / (pred_t_max - pred_t_min + 1e-9)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Define metrics to plot with normalization (only energy is demonstrated below)
    metrics = [
        ("energy_norm", "Normalized Energy per Token", "predicted_energy_norm", "actual_energy_norm"),
        ("power_norm", "Normalized Power Consumption", "predicted_power_norm", "actual_power_norm"),
        ("time_norm", "Normalized Execution Time", "predicted_time_norm", "actual_time_norm")
    ]
    
    # Generate comparative normalized plots (combined actual and predicted)
    for metric, ylabel, pred_col, actual_col in metrics:
        plt.figure(figsize=(12, 6))
        plot_df = df.melt(
            id_vars=["block_dims", "seq_len"],
            value_vars=[pred_col, actual_col],
            var_name="type",
            value_name=metric
        )
        # Replace type names for clarity
        plot_df["type"] = plot_df["type"].map({pred_col: "Predicted", actual_col: "Actual"})
        
        ax = sns.lineplot(
            data=plot_df,
            x="block_dims",
            y=metric,
            hue="seq_len",
            style="type",
            markers=True,
            palette="tab10",
            estimator=None
        )
        # plt.title(f"{ylabel} - Predicted vs. Actual")
        plt.xticks(rotation=45, fontsize=14)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks[::4])
        plt.xlabel("Block Dimensions", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.legend(bbox_to_anchor=(0.05, 0.95), loc="upper left", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/combined_norm_{metric}.png", dpi=150)
        plt.close()


def extract_rq3_summary(csv_path, out_txt="rq3_summary.txt"):
    df = pd.read_csv(csv_path)
    valid_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_x"].isin(valid_blocks) &
        df["block_y"].isin(valid_blocks)
    ].copy()
    
    # Compute energy in Joules: (W * ns) / 1e9
    df['energy'] = (df['actual_power'] * df['actual_time_ns']) / 1e9
    
    seq_lens = sorted(df["seq_len"].unique())
    total_configs = len(valid_blocks) ** 2  # 121 configs per seq_len
    
    summary = []
    txt = []
    txt.append("RQ3 Summary Statistics\n======================\n")
    txt.append(f"Total valid configs per seq_len: {total_configs}\n")
    txt.append("")
    txt.append(f"{'Seq. Length':>10} | {'Total Configs':>13} | {'Recommended':>11} | {'CRR':>6}")
    txt.append("-"*50)
    
    avg_recommended = []
    avg_crr = []
    pareto_counts = []
    config_counts = []

    for seq in seq_lens:
        dseq = df[df["seq_len"] == seq]
        config_counts.append(len(dseq))
        
        # Compute Pareto front (minimize time and energy)
        points = []
        for idx, row in dseq.iterrows():
            points.append((row['actual_time_ns'], row['energy'], idx))
        
        # Sort by time (ascending) and energy (ascending)
        sorted_points = sorted(points, key=lambda x: (x[0], x[1]))
        pareto_indices = []
        min_energy = float('inf')
        
        for t, e, idx in sorted_points:
            if e < min_energy:
                pareto_indices.append(idx)
                min_energy = e
                
        n_pareto = len(pareto_indices)
        pareto_counts.append(n_pareto)
        crr = 1 - n_pareto / total_configs
        
        avg_recommended.append(n_pareto)
        avg_crr.append(crr)
        txt.append(f"{seq:>10} | {total_configs:>13} | {n_pareto:>11} | {crr:>6.3f}")

    # Calculate averages from actual data
    avg_pareto = np.mean(pareto_counts)
    avg_crr_val = np.mean(avg_crr)
    avg_energy_per_run = df['energy'].mean()
    flipflop_configs = 5  # assumed based on methodology
    
    # Resource savings calculations
    profiling_time_per_config = 6  # 5 runs x 1.2s
    total_profiling_time_per_seq = total_configs * profiling_time_per_config
    flipflop_time_per_seq = flipflop_configs * profiling_time_per_config
    time_speedup = total_profiling_time_per_seq / flipflop_time_per_seq
    
    # Energy savings calculations
    skipped_configs = total_configs - flipflop_configs
    energy_saved_per_seq = 5 * skipped_configs * avg_energy_per_run
    total_energy_saved = 8 * energy_saved_per_seq
    
    # Developer impact calculations
    bf_time_5seq = 5 * total_profiling_time_per_seq / 3600
    ff_time_5seq = 5 * flipflop_time_per_seq / 3600
    bf_time_100k = 100 * total_profiling_time_per_seq / 3600
    ff_time_100k = 100 * flipflop_time_per_seq / 3600
    
    # Energy impact calculations (1 kWh = 3.6e6 J)
    energy_saved_100k = 100 * skipped_configs * 5 * avg_energy_per_run
    energy_kwh_100k = energy_saved_100k / 3.6e6
    co2_100k = energy_kwh_100k * 0.5  # 0.5 kg CO2 per kWh
    
    # Case study calculations
    traditional_time_20k = 20 * total_profiling_time_per_seq / 3600
    ff_time_20k = 20 * flipflop_time_per_seq / 3600
    energy_saved_20k = 20 * skipped_configs * 5 * avg_energy_per_run
    co2_20k = (energy_saved_20k / 3.6e6) * 0.5
    
    # Generate report
    txt.append("-"*50)
    txt.append(f"{'Avg.':>10} | {total_configs:>13} | {avg_pareto:>11.1f} | {avg_crr_val:>6.3f}")
    txt.append("")
    
    txt.append("Resource Savings\n----------------")
    txt.append(f"Profiling time per config: {profiling_time_per_config}s (5 runs x 1.2s)")
    txt.append(f"Total profiling time per seq_len: {total_configs} x 6s = {total_profiling_time_per_seq/60:.1f} min")
    txt.append(f"For 8 seq_lens: {8*total_profiling_time_per_seq/3600:.1f} hours")
    txt.append(f"FlipFlop time: {flipflop_configs} configs x 6s = {flipflop_time_per_seq/60:.1f} min per seq_len")
    txt.append(f"\nTime savings: {total_profiling_time_per_seq/60:.1f} min → {flipflop_time_per_seq/60:.1f} min ({time_speedup:.1f}x faster)")
    txt.append("")
    
    txt.append("Energy Savings\n--------------")
    txt.append(f"Avg. energy per run (actual): {avg_energy_per_run*1000:.2f} mJ")
    txt.append(f"Skipped configs per seq_len: {skipped_configs}")
    txt.append(f"Energy saved per seq_len: 5 runs x {skipped_configs} configs x {avg_energy_per_run*1000:.2f}mJ = {energy_saved_per_seq:.0f}J")
    txt.append(f"Total energy saved (8 seq_lens): {total_energy_saved:.0f}J = {total_energy_saved/1000:.3f} kJ")
    txt.append("")
    
    txt.append("Developer Impact\n----------------")
    txt.append(f"Iterative tuning: 5 seq_lens in {ff_time_5seq*60:.1f} min vs. {bf_time_5seq:.1f} hours ({bf_time_5seq/ff_time_5seq:.1f}x faster)")
    txt.append(f"Scaling: 100 kernels, brute-force: {bf_time_100k:.1f}h → FlipFlop: {ff_time_100k:.1f}h ({bf_time_100k/ff_time_100k:.1f}x faster)")
    txt.append(f"Energy-efficient: ~{energy_kwh_100k:.1f} kWh saved per 100 kernels ({co2_100k:.1f} kg CO2)")
    txt.append("")
    
    txt.append("Case Study: LLaMA-3-8B\n----------------------")
    txt.append(f"Traditional: {traditional_time_20k:.1f}h for 20 kernels")
    txt.append(f"FlipFlop: {ff_time_20k:.1f}h optimization ({traditional_time_20k/ff_time_20k:.1f}x faster)")
    txt.append(f"Carbon: ~{co2_20k:.1f} kg CO2 avoided")
    
    with open(out_txt, "w") as f:
        f.write("\n".join(txt))

if __name__ == "__main__":
    plot_rq3_analysis(
        csv_path="rq3_data/full_energy_results.csv",
        out_dir="plots"
    )

    plot_rq3_analysis_normalized(
        csv_path="rq3_data/energy_model_results_20250414_173438.csv",
        out_dir="plots_norm"
    )

    extract_rq3_summary(
        csv_path="rq3_data/energy_model_results_20250414_173438.csv",
        out_txt="rq3_summary.txt"
    )