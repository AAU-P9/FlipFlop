import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def filter_and_plot_rq1_multi_seq_len(csv_file, out_dir="plots_rq1"):
    """
    Reads 'csv_file', filters only configurations where block_size_x,y ∈ {1,2,4,8,16,32,64,128,256,512,1024},
    sorts them by total thread count (ascending) and block_size_x (ascending),
    then creates a single line-plot of Joules/token vs. block_dims, 
    with a separate line for each distinct seq_len on the same plot.
    """
    # 1. Read data
    df = pd.read_csv(csv_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # 2. Filter for valid block_size_x, block_size_y
    valid_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_size_x"].isin(valid_vals) &
        df["block_size_y"].isin(valid_vals)
    ].copy()

    # 3. Compute total threads and block_dims label
    df["threads_per_block"] = df["block_size_x"] * df["block_size_y"]
    df["block_dims"] = df.apply(lambda row: f"{row['block_size_x']}x{row['block_size_y']}", axis=1)

    # 4. Sort by (threads_per_block, block_size_x, block_size_y)
    df["sort_key"] = df.apply(
        lambda row: (row["threads_per_block"], row["block_size_x"], row["block_size_y"]), 
        axis=1
    )
    df.sort_values("sort_key", inplace=True)

    # 5. Convert block_dims to a categorical with the sorted order
    unique_block_dims = df["block_dims"].unique().tolist()
    df["block_dims"] = pd.Categorical(df["block_dims"], categories=unique_block_dims, ordered=True)

    # 6. Prepare the figure: Joules/token vs block_dims, hue=seq_len
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x="block_dims",
        y="Joules/token",
        hue="seq_len",
        marker="o",
        sort=False,  # do not re-sort the categorical x-axis,
        palette="tab20",  # Higher contrast palette
        linewidth=1.5
    )

    unique_block_dims = df["block_dims"].cat.categories.tolist()
    n = 2  
    xticks_indices = np.arange(len(unique_block_dims))[::n]
    xticks_labels = unique_block_dims[::n]

    # 7. Final formatting
    plt.xticks(
        ticks=xticks_indices,
        labels=xticks_labels,
        rotation=45,  # Vertical labels
        ha='center',   # Center-aligned
        fontsize=8     # Smaller font
    )


    plt.xlabel("Block Dimensions")
    plt.ylabel("Joules/token")
    plt.title("Energy per Token vs Block Dimensions (One line per seq_len)")
    plt.legend(title="Sequence Length", bbox_to_anchor=(0.05, 0.95), loc="upper left")

    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 8. Save and finish
    outname = os.path.join(out_dir, "energy_vs_block_dims_multi_seq_len.png")
    plt.savefig(outname, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to: {outname}")

def filter_and_plot_rq1_multi_seq_len_vs_occupancy(csv_file, out_dir="plots_rq1"):
    """
    Modified version with secondary y-axis showing occupancy percentage.
    Requires occupancy_dict: {thread_count: occupancy_percentage}
    """
    # 1. Read data
    df = pd.read_csv(csv_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    occupancy_dict = {
        1: 50.0, 2: 50.0, 4: 50.0, 8: 50.0,
        16: 50.0, 32: 50.0, 64: 100.0, 128: 100.0,
        256: 100.0, 512: 100.0, 1024: 66.0
    }

    # 2. Filter for valid block_size_x, block_size_y
    valid_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_size_x"].isin(valid_vals) &
        df["block_size_y"].isin(valid_vals)
    ].copy()

    # 3. Compute derived columns
    df["threads_per_block"] = df["block_size_x"] * df["block_size_y"]
    df["block_dims"] = df.apply(lambda row: f"{row['block_size_x']}x{row['block_size_y']}", axis=1)

    # 4. Sort data
    df["sort_key"] = df.apply(
        lambda row: (row["threads_per_block"], row["block_size_x"], row["block_size_y"]), 
        axis=1
    )
    df.sort_values("sort_key", inplace=True)

    # 5. Create categorical ordering
    unique_block_dims = df["block_dims"].unique().tolist()
    df["block_dims"] = pd.Categorical(df["block_dims"], categories=unique_block_dims, ordered=True)

    # 6. Create main plot
    plt.figure(figsize=(12, 7))
    ax1 = plt.gca()
    
    sns.lineplot(
        data=df,
        x="block_dims",
        y="Joules/token",
        hue="seq_len",
        marker="o",
        sort=False,
        palette="tab20",
        linewidth=2.5,
        ax=ax1
    )

    # 7. Configure x-axis
    n = 2  
    xticks_indices = np.arange(len(unique_block_dims))[::n]
    xticks_labels = unique_block_dims[::n]

    ax1.set_xticks(xticks_indices)
    ax1.set_xticklabels(
        xticks_labels,
        rotation=45,
        ha='center',
        fontsize=16
    )

    ax1.set_xlabel("Block Dimensions (Width × Height)", fontsize=20)
    ax1.set_ylabel("Energy Consumption (Joules/token)", fontsize=20)
    ax1.grid(alpha=0.3)
    # set tick font size
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # 8. Create secondary y-axis for occupancy
    ax2 = ax1.twinx()
    
    # Calculate occupancy percentages from precomputed dictionary
    x_indices = np.arange(len(unique_block_dims))
    occupancies = [occupancy_dict[int(bd.split('x')[0]) * int(bd.split('x')[1])] 
                   for bd in unique_block_dims]

    ax2.plot(
        x_indices, 
        occupancies,
        color='red', 
        linestyle='--', 
        marker='x',
        markersize=10,
        linewidth=3.5,
        label='SM Occupancy'
    )

    ax2.set_ylim(0, 105)
    ax2.set_ylabel("SM Occupancy (%)", color='dimgray', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='gray', labelsize=18)

    # 9. Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        title="Seq_Length & Occup.",
        bbox_to_anchor=(0.01, 0.99),
        loc="upper left"
    )

    # plt.title("Energy Efficiency vs Block Configuration with SM Occupancy", fontsize=16)
    plt.tight_layout()

    # 10. Save output
    out_path = os.path.join(out_dir, "energy_vs_block_dims_with_occupancy.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {out_path}")

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
    plt.xlabel("block_dims", fontsize=14)
    plt.ylabel("Joules/token", fontsize=14)
    plt.title("Energy per Token (J) vs Block Dimensions")
    plt.legend(title="Threads/Block", bbox_to_anchor=(0.05, 0.95), loc="upper left")
    plt.tight_layout()

    outpath = os.path.join(out_dir, "energy_vs_block_dims.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to: {outpath}")



def filter_and_plot_rq1_multi_seq_len_vs_occupancy_baseline(csv_file, out_dir="plots_rq1"):
    """
    Plots Joules/token vs. block_dims for multiple seq_len lines,
    adds a horizontal reference line for the occupancy-based configuration (e.g., 256x1),
    and, for seq_len 128 (and thread count 512), draws horizontal lines at the min, max, and mean energy values,
    with arrows indicating the energy gaps.
    """
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Read data
    df = pd.read_csv(csv_file)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Filter for valid block_size_x, block_size_y
    valid_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_size_x"].isin(valid_vals) &
        df["block_size_y"].isin(valid_vals)
    ].copy()

    # 3. Compute total threads and block_dims label
    df["threads_per_block"] = df["block_size_x"] * df["block_size_y"]
    df["block_dims"] = df.apply(lambda row: f"{row['block_size_x']}x{row['block_size_y']}", axis=1)

    # 4. Sort by (threads_per_block, block_size_x, block_size_y)
    df["sort_key"] = df.apply(
        lambda row: (row["threads_per_block"], row["block_size_x"], row["block_size_y"]),
        axis=1
    )
    df.sort_values("sort_key", inplace=True)

    # 5. Create categorical ordering for block_dims on x-axis
    unique_block_dims = df["block_dims"].unique().tolist()
    df["block_dims"] = pd.Categorical(df["block_dims"], categories=unique_block_dims, ordered=True)

    # 6. Start the figure
    plt.figure(figsize=(10,6))
    ax = plt.gca()

    sns.lineplot(
        data=df,
        x="block_dims",
        y="Joules/token",
        hue="seq_len",
        marker="o",
        sort=False,  # Do not re-sort categorical x-axis
        palette="tab20",
        linewidth=1.5,
        ax=ax
    )

    # 7. Set x-tick labels to show every 2nd tick with increased font size.
    unique_block_dims = df["block_dims"].cat.categories.tolist()
    n = 2  
    xticks_indices = np.arange(len(unique_block_dims))[::n]
    xticks_labels = [unique_block_dims[i] for i in xticks_indices]

    ax.set_xticks(xticks_indices)
    ax.set_xticklabels(
        xticks_labels,
        rotation=45,
        ha='center',
        fontsize=16
    )

    ax.set_xlabel("Block Dimensions", fontsize=20)
    ax.set_ylabel("Energy (Joules/token)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(alpha=0.3)

    # 8. Add horizontal reference line for the occupancy-based configuration.
    # Assume occupancy-based configuration is block_size_x=256 and block_size_y=1.
    # occ_x, occ_y = 256, 1
    # df_occ = df[(df["block_size_x"] == occ_x) & (df["block_size_y"] == occ_y)]
    # if not df_occ.empty:
    #     occ_energy = df_occ["Joules/token"].mean()
    #     ax.axhline(
    #         y=occ_energy,
    #         color='blue',
    #         linestyle=':',
    #         linewidth=2.0,
    #         label=f"Occ. Config ({occ_x}×{occ_y}) Energy"
    #     )

    # 9. For sequence length 128 and total threads 512, add horizontal lines for min, max, and mean energy.
    df_128 = df[(df["seq_len"] == 128) & (df["threads_per_block"] == 512)]
    if not df_128.empty:
        energy_min = df_128["Joules/token"].min()
        energy_max = df_128["Joules/token"].max()
        energy_mean = df_128["Joules/token"].mean()
        ax.axhline(y=energy_min, color='green', linestyle='--', linewidth=2.0, label="Min Energy Config (x1, y1), seq=128")
        ax.axhline(y=energy_max, color='magenta', linestyle='--', linewidth=2.0, label="Max Energy Config (x3, y3) (seq=128)")
        ax.axhline(y=energy_mean, color='orange', linestyle='-.', linewidth=2.0, label="Occupancy-Based Config (x2, y2), seq=128")
        
        # Add an arrow between min and max
        ax.annotate(
            "",
            xy=(0.5, energy_min),
            xytext=(0.5, energy_max),
            xycoords=('axes fraction', 'data'),
            arrowprops=dict(arrowstyle="<->", color='black', linewidth=2)
        )
        gap_max_min = energy_max - energy_min
        ax.text(
            0.32, (energy_min + energy_max) / 2,
            f"Δ={gap_max_min:.2e}",
            transform=ax.get_yaxis_transform(),
            fontsize=16,
            color='black',
            va='center'
        )
        # Add an arrow between mean and min
        ax.annotate(
            "",
            xy=(0.20, energy_min),
            xytext=(0.20, energy_mean),
            xycoords=('axes fraction', 'data'),
            arrowprops=dict(arrowstyle="<->", color='blue', linewidth=2)
        )
        gap_mean_min = energy_mean - energy_min
        ax.text(
            0.03, (energy_min + energy_mean) / 2,
            f"Δ={gap_mean_min:.2e}",
            transform=ax.get_yaxis_transform(),
            fontsize=16,
            color='blue',
            va='center'
        )

    # 10. Add legend with two columns
    ax.legend(title="Seq Length", bbox_to_anchor=(0.05, 0.95), loc="upper left", ncol=2)
    plt.tight_layout()

    # 11. Save output
    out_path = os.path.join(out_dir, "energy_vs_block_dims_with_occupancy_baseline.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    # Example usage:
    csv_file = "rq1_data/mha_tuning_results.csv"  # Modify as needed
    # filter_and_plot_rq1(csv_file, out_dir="rq1_plots")
    # filter_and_plot_rq1_multi_seq_len(csv_file, out_dir="rq1_plots")
    # filter_and_plot_rq1_multi_seq_len_vs_occupancy(csv_file, out_dir="rq1_plots")
    filter_and_plot_rq1_multi_seq_len_vs_occupancy_baseline(csv_file, out_dir="rq1_plots")
