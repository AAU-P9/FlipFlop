import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def filter_and_plot_rq1_multi_seq_len(csv_file, out_dir="plots_rq1"):
    """
    Reads 'csv_file', filters configurations where block_size_x and block_size_y 
    are in {1,2,4,8,16,32,64,128,256,512,1024}, sorts them by total thread count (ascending) 
    and block_size_x (ascending), then creates a single plot of Joules/token vs. block_dims.
    
    For each sequence length (seq_len), separate lines are drawn for each distinct threads_per_block
    group; however, the lines are plotted in the same color for a given seq_len (using the "Dark2" palette)
    and only one legend entry per seq_len is shown (legend placed inside top-left).
    """
    # 1. Read CSV data
    df = pd.read_csv(csv_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # 2. Filter for valid block sizes
    valid_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_size_x"].isin(valid_vals) & 
        df["block_size_y"].isin(valid_vals)
    ].copy()
    
    # 3. Compute total threads and block_dims label
    df["threads_per_block"] = df["block_size_x"] * df["block_size_y"]
    df["block_dims"] = df.apply(lambda row: f"{row['block_size_x']}x{row['block_size_y']}", axis=1)
    
    # 4. Sort by total threads, then by block_size_x then block_size_y
    df["sort_key"] = df.apply(lambda row: (row["threads_per_block"], row["block_size_x"], row["block_size_y"]), axis=1)
    df.sort_values("sort_key", inplace=True)
    
    # 5. Convert block_dims to an ordered categorical variable (preserving sorted order)
    unique_block_dims = df["block_dims"].unique().tolist()
    df["block_dims"] = pd.Categorical(df["block_dims"], categories=unique_block_dims, ordered=True)
    # Create numeric x-axis values from categorical codes
    df["x_pos"] = df["block_dims"].cat.codes

    # 6. Prepare color palette: one distinct color per sequence length.
    seq_lens = sorted(df["seq_len"].unique())
    palette = sns.color_palette("Dark2", n_colors=len(seq_lens))
    seq_len_colors = dict(zip(seq_lens, palette))
    
    plt.figure(figsize=(12,6))
    
    # 7. Plot disconnected lines:
    #    For each group defined by (seq_len, threads_per_block), plot the line segment.
    #    Only label the line once per seq_len.
    labeled_seq = set()
    groups = df.groupby(["seq_len", "threads_per_block"])
    for (seq_len, threads), group in groups:
        group = group.sort_values("x_pos")
        color = seq_len_colors.get(seq_len, "black")
        # Only add label for the first occurrence for each seq_len
        label = f"{seq_len}" if seq_len not in labeled_seq else None
        if label is not None:
            labeled_seq.add(seq_len)
        plt.plot(
            group["x_pos"],
            group["Joules/token"],
            linestyle="-",
            color=color,
            label=label,
            marker="o",
            markersize=4
        )
    
    # 8. Set x-axis ticks and labels using the sorted order of block_dims
    plt.xticks(ticks=range(len(unique_block_dims)), labels=unique_block_dims, rotation=45, ha="right", fontsize=10)
    plt.xlabel("Block Dimensions (block_size_x x block_size_y)", fontsize=14)
    plt.ylabel("Joules/token", fontsize=14)
    plt.title("Energy per Token vs. Block Dimensions")
    
    # 9. Place the legend inside the plot (top-left)
    plt.legend(title="Sequence Length", loc="upper left", bbox_to_anchor=(0.05, 0.95))
    plt.tight_layout()
    
    outpath = os.path.join(out_dir, "energy_vs_block_dims_disconnected.png")
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot to: {outpath}")

def filter_and_plot_rq1(csv_file, out_dir="plots"):
    """
    Reads 'csv_file', filters only configurations where block_size_x,y ∈ {1,2,4,8,16,32,64,128,256,512,1024},
    sorts them by total thread count and then by block_size_x, and plots Joules/token in a lineplot
    with the x-axis = block_dims (e.g., "2x16") and hue = threads_per_block.

    The final figure is saved as 'energy_vs_block_dims.png' in 'out_dir'.
    """

    # 1. Read data
    df = pd.read_csv(csv_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 2. Filter for valid block_size_x, block_size_y in the specified set
    valid_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_size_x"].isin(valid_vals) &
        df["block_size_y"].isin(valid_vals)
    ].copy()

    # 3. Compute total threads and block_dims string
    df["threads_per_block"] = df["block_size_x"] * df["block_size_y"]
    df["block_dims"] = df.apply(lambda row: f"{row['block_size_x']}x{row['block_size_y']}", axis=1)

    # 4. Sort by (threads_per_block, block_size_x, block_size_y)
    df["sort_key"] = df.apply(lambda row: (row["threads_per_block"], row["block_size_x"], row["block_size_y"]), axis=1)
    df.sort_values("sort_key", inplace=True)

    # 5. Convert 'block_dims' to a categorical with the sorted order
    unique_block_dims = df["block_dims"].unique().tolist()
    df["block_dims"] = pd.Categorical(df["block_dims"], categories=unique_block_dims, ordered=True)

    # 6. Plot Joules/token vs. block_dims, line plot with hue = threads_per_block
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x="block_dims",
        y="Joules/token",
        hue="threads_per_block",
        marker="o",
        sort=False,   # Do not re-sort x-axis categories
        palette="viridis"
    )

    plt.xticks(rotation=45, ha="right")
    plt.xlabel("block_dims")
    plt.ylabel("Joules/token")
    plt.title("Energy per Token (J) vs Block Dimensions")
    plt.legend(title="Threads/Block", bbox_to_anchor=(0.05, 0.95), loc="upper left")
    plt.tight_layout()

    outpath = os.path.join(out_dir, "energy_vs_block_dims.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to: {outpath}")

if __name__ == "__main__":
    # Example usage:
    csv_file = "rq1_data/mha_tuning_results.csv"  # Modify as needed
    filter_and_plot_rq1(csv_file, out_dir="rq1_plots")
    filter_and_plot_rq1_multi_seq_len(csv_file, out_dir="rq1_plots")
