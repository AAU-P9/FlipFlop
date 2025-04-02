import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

if __name__ == "__main__":
    plot_rq3_analysis(
        csv_path="rq3_data/full_energy_results.csv",
        out_dir="plots"
    )