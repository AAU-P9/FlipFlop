import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_rq4_analysis(csv_path, out_dir="rq4_plots"):
    """Generate comparative plots for RQ4 energy model validation."""
    # Load CSV
    df = pd.read_csv(csv_path)

    # Filter valid block configurations for convenience
    valid_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_x"].isin(valid_blocks) &
        df["block_y"].isin(valid_blocks)
    ].copy()

    # Create composite fields for easier labeling
    df["block_dims"] = df["block_x"].astype(str) + "x" + df["block_y"].astype(str)
    df["threads_per_block"] = df["block_x"] * df["block_y"]
    # Sort by (thread_count, block_x) so x-axis is in ascending order
    df = df.sort_values(["threads_per_block", "block_x"])
    # Make block_dims a categorical with the encountered order
    df["block_dims"] = pd.Categorical(df["block_dims"], ordered=True, categories=df["block_dims"].unique())

    # Compute predicted vs. actual energy per token
    #   predicted_energy[J/token] = predicted_power[W] * predicted_time_ns[s]/(beamsize*n_steps)
    #   (watch out for nanosec -> sec)
    df["predicted_energy"] = (
        df["predicted_power"] * (df["predicted_time_ns"] / 1e9)
        / (df["beamsize"] * df["n_steps"])
    )
    df["actual_energy"] = (
        df["actual_power"] * (df["actual_time_ns"] / 1e9)
        / (df["beamsize"] * df["n_steps"])
    )
    # Convert time from ns -> seconds
    df["predicted_time_s"] = df["predicted_time_ns"] / 1e9
    df["actual_time_s"] = df["actual_time_ns"] / 1e9

    # Make sure our output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # We will produce three sets of plots: {energy, power, time}
    # with predicted vs actual lines, as well as separate predicted-only and actual-only lines.
    metrics = [
        ("energy", "Energy per Token (Joules/token)", "predicted_energy", "actual_energy"),
        ("power", "Power (Watts)", "predicted_power", "actual_power"),
        ("time", "Execution Time (seconds)", "predicted_time_s", "actual_time_s"),
    ]

    for metric, ylabel, pred_col, actual_col in metrics:
        # Combined plot: predicted vs. actual lines on the same axes
        plt.figure(figsize=(12, 6))
        melted = df.melt(
            id_vars=["block_dims", "n_steps"],
            value_vars=[pred_col, actual_col],
            var_name="type",
            value_name=metric
        )
        sns.lineplot(
            data=melted,
            x="block_dims",
            y=metric,
            hue="n_steps",     # color by n_steps
            style="type",      # line style by predicted/actual
            markers=True,
            estimator=None,
            palette="tab10"
        )
        plt.title(f"{ylabel} — Predicted vs Actual")
        plt.xlabel("Block Dimensions")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/combined_{metric}.png", dpi=150)
        plt.close()

        # Predicted-only plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="block_dims",
            y=pred_col,
            hue="n_steps",
            marker="o",
            palette="tab10"
        )
        plt.title(f"Predicted {ylabel}")
        plt.xlabel("Block Dimensions")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/predicted_{metric}.png", dpi=150)
        plt.close()

        # Actual-only plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="block_dims",
            y=actual_col,
            hue="n_steps",
            marker="o",
            palette="tab10"
        )
        plt.title(f"Actual {ylabel}")
        plt.xlabel("Block Dimensions")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/actual_{metric}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    # Example usage:
    # python plot_rq4_analysis.py
    csv_file = "rq4_data/llama3_energy_results_20250403_103925.csv"
    out_dir = "rq4_plots"
    plot_rq4_analysis(csv_file, out_dir)
