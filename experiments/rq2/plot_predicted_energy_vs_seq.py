import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import seaborn as sns

def plot_predicted_energy_vs_shape(csv_path, out_path="predicted_energy_vs_shape.png"):
    df = pd.read_csv(csv_path)
    # Try to infer batch_size and seq_len if not present
    if "batch_size" not in df.columns:
        df["batch_size"] = 1  # fallback
    if "seq_len" not in df.columns:
        raise ValueError("CSV must contain a 'seq_len' column.")
    # Compute predicted energy per token if not present
    if "predicted_energy" not in df.columns:
        if "predicted_power" in df.columns and "predicted_time_ns" in df.columns:
            df["predicted_energy"] = (
                df["predicted_power"] * df["predicted_time_ns"] / 1e9
            ) / (df["batch_size"] * df["seq_len"])
        else:
            raise ValueError("CSV must contain 'predicted_energy' or ('predicted_power' and 'predicted_time_ns') columns.")
    # Create block shape label
    df["block_shape"] = df.apply(lambda row: f"{int(row['block_x'])}x{int(row['block_y'])}", axis=1)
    # Sort by thread count, then block_x, then block_y
    df["thread_count"] = df["block_x"] * df["block_y"]
    df = df.sort_values(["thread_count", "block_x", "block_y"])
    # Get unique block shapes in sorted order
    block_shapes_sorted = df.drop_duplicates(["block_shape"])[["thread_count", "block_x", "block_y", "block_shape"]]
    block_shapes_sorted = block_shapes_sorted.sort_values(["thread_count", "block_x", "block_y"])
    block_shape_labels = block_shapes_sorted["block_shape"].tolist()
    # Plot
    plt.figure(figsize=(16, 12))
    seq_lens = sorted(df["seq_len"].unique())
    for seq_len in seq_lens:
        sub = df[df["seq_len"] == seq_len]
        x = block_shape_labels
        y = [sub[sub["block_shape"] == shape]["predicted_energy"].values[0] if shape in sub["block_shape"].values else np.nan for shape in x]
        plt.plot(x, y, marker="o", label=seq_len)
    plt.yscale("log")
    plt.xlabel("Block Shape (X x Y)", fontsize=24)
    plt.ylabel("Predicted Energy per Token (Joules/token)", fontsize=24)
    # Only show every other x label, but keep all ticks
    ax = plt.gca()
    ticks = np.arange(len(block_shape_labels))
    ax.set_xticks(ticks)
    # Show only alternate labels, make them large
    new_labels = [label if i % 2 == 0 else "" for i, label in enumerate(block_shape_labels)]
    ax.set_xticklabels(new_labels, rotation=60, fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    # Add dark vertical grid lines for each x-tick
    # for tick in ticks:
    #     ax.axvline(x=tick, color='#333333', linestyle='-', linewidth=1.2, alpha=0.1, zorder=0)
    # plt.grid(True, which="both", axis="y", ls="--")
    plt.legend(title="Sequence Length", fontsize=18, title_fontsize=20)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

def plot_actual_energy_vs_shape(csv_path, out_path="actual_energy_vs_shape.png", seq_len_param=None):
    df = pd.read_csv(csv_path)
    if "batch_size" not in df.columns:
        df["batch_size"] = 1  # fallback
    if "seq_len" not in df.columns:
        raise ValueError("CSV must contain a 'seq_len' column.")
    if "actual_energy" not in df.columns:
        if "actual_power" in df.columns and "actual_time_ns" in df.columns:
            df["actual_energy"] = (
                df["actual_power"] * df["actual_time_ns"] / 1e9
            ) / (df["batch_size"] * df["seq_len"])
        else:
            raise ValueError("CSV must contain 'actual_energy' or ('actual_power' and 'actual_time_ns') columns.")
    df["block_shape"] = df.apply(lambda row: f"{int(row['block_x'])}x{int(row['block_y'])}", axis=1)
    df["thread_count"] = df["block_x"] * df["block_y"]
    df = df.sort_values(["thread_count", "block_x", "block_y"])
    block_shapes_sorted = df.drop_duplicates(["block_shape"])[["thread_count", "block_x", "block_y", "block_shape"]]
    block_shapes_sorted = block_shapes_sorted.sort_values(["thread_count", "block_x", "block_y"])
    block_shape_labels = block_shapes_sorted["block_shape"].tolist()
    plt.figure(figsize=(16, 12))
    # Only plot for the given sequence length if specified
    seq_lens_to_plot = [seq_len_param] if seq_len_param is not None else sorted(df["seq_len"].unique())
    for seq_len in seq_lens_to_plot:
        sub = df[df["seq_len"] == seq_len]
        if sub.empty:
            print(f"No data for sequence length {seq_len}")
            continue
        x = block_shape_labels
        y = [sub[sub["block_shape"] == shape]["actual_energy"].values[0] if shape in sub["block_shape"].values else np.nan for shape in x]
        plt.plot(x, y, marker="o", label=seq_len)
    plt.yscale("log")
    plt.xlabel("Block Shape (X x Y)", fontsize=24)
    plt.ylabel("Actual Energy per Token (Joules/token)", fontsize=24)
    ax = plt.gca()
    ticks = np.arange(len(block_shape_labels))
    ax.set_xticks(ticks)
    new_labels = [label if i % 2 == 0 else "" for i, label in enumerate(block_shape_labels)]
    ax.set_xticklabels(new_labels, rotation=60, fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    plt.legend(title="Sequence Length", fontsize=18, title_fontsize=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

# Example usage for seq_len_param=8192:
# plot_actual_energy_vs_shape('your_csv.csv', 'your_out_path', seq_len_param=8192)

def plot_rq3_analysis_normalized(csv_path, out_dir="rq3_plots_norm", seq_len_param=None):
    df = pd.read_csv(csv_path)
    if "batch_size" not in df.columns:
        df["batch_size"] = 1  # fallback
    if "predicted_energy" not in df.columns:
        if "predicted_power" in df.columns and "predicted_time_ns" in df.columns:
            df["predicted_energy"] = (
                df["predicted_power"] * df["predicted_time_ns"] / 1e9
            ) / (df["batch_size"] * df["seq_len"])
        else:
            raise ValueError("CSV must contain 'predicted_energy' or ('predicted_power' and 'predicted_time_ns') columns.")
    if "actual_energy" not in df.columns:
        if "actual_power" in df.columns and "actual_time_ns" in df.columns:
            df["actual_energy"] = (
                df["actual_power"] * df["actual_time_ns"] / 1e9
            ) / (df["batch_size"] * df["seq_len"])
        else:
            raise ValueError("CSV must contain 'actual_energy' or ('actual_power' and 'actual_time_ns') columns.")
    valid_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    df = df[
        df["block_x"].isin(valid_blocks) &
        df["block_y"].isin(valid_blocks)
    ].copy()
    df["block_dims"] = df["block_x"].astype(str) + "x" + df["block_y"].astype(str)
    df["threads_per_block"] = df["thread_count"]
    df = df.sort_values(["seq_len", "threads_per_block", "block_x"])
    df["block_dims"] = pd.Categorical(df["block_dims"], ordered=True, categories=df["block_dims"].unique())

    # Normalize predicted and actual energy per token for each sequence length
    df["predicted_energy_norm"] = np.nan
    df["actual_energy_norm"] = np.nan
    for seq_len in df["seq_len"].unique():
        mask = df["seq_len"] == seq_len
        pred = df.loc[mask, "predicted_energy"]
        act = df.loc[mask, "actual_energy"]
        pred_min, pred_max = pred.min(), pred.max()
        act_min, act_max = act.min(), act.max()
        df.loc[mask, "predicted_energy_norm"] = (pred - pred_min) / (pred_max - pred_min + 1e-9)
        df.loc[mask, "actual_energy_norm"] = (act - act_min) / (act_max - act_min + 1e-9)

    os.makedirs(out_dir, exist_ok=True)
    # Only plot for the given sequence length if specified
    seq_lens_to_plot = [seq_len_param] if seq_len_param is not None else sorted(df["seq_len"].unique())
    for seq_len in seq_lens_to_plot:
        sub = df[df["seq_len"] == seq_len]
        if sub.empty:
            print(f"No data for sequence length {seq_len}")
            continue
        x = sub["block_dims"]
        y_pred = sub["predicted_energy_norm"]
        y_act = sub["actual_energy_norm"]
        plt.figure(figsize=(16, 10))
        plt.plot(x, y_pred, marker="o", label="Predicted")
        plt.plot(x, y_act, marker="x", linestyle="--", label="Actual")
        # Only show alternate x labels, but keep all ticks
        ax = plt.gca()
        ticks = np.arange(len(x))
        ax.set_xticks(ticks)
        new_labels = [label if i % 2 == 0 else "" for i, label in enumerate(x)]
        ax.set_xticklabels(new_labels, rotation=60, fontsize=22)
        plt.yticks(fontsize=18)
        plt.xlabel("Block Shape (X x Y)", fontsize=24)
        plt.ylabel("Normalized Energy per Token", fontsize=24)
        plt.legend(title="seq_len=8192", fontsize=20, title_fontsize=20)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"normalized_energy_per_token_seq{seq_len}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved normalized plot for seq_len={seq_len} to {out_path}")

        # --- Metric: Compare local maxima and minima locations ---
        if not sub.empty:
            pred_vals = y_pred.values
            act_vals = y_act.values
            block_labels = list(x)

            def find_local_extrema(arr, mode="max"):
                extrema = []
                for i in range(1, len(arr)-1):
                    if mode == "max" and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                        extrema.append(i)
                    elif mode == "min" and arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                        extrema.append(i)
                return extrema

            pred_maxima = find_local_extrema(pred_vals, mode="max")
            pred_minima = find_local_extrema(pred_vals, mode="min")
            act_maxima = find_local_extrema(act_vals, mode="max")
            act_minima = find_local_extrema(act_vals, mode="min")

            def has_nearby(idx, idx_list, window=1):
                return any(abs(idx - j) <= window for j in idx_list)

            # For each predicted extremum, check if actual has one nearby
            max_matches = sum(has_nearby(i, act_maxima) for i in pred_maxima)
            min_matches = sum(has_nearby(i, act_minima) for i in pred_minima)
            max_total = len(pred_maxima)
            min_total = len(pred_minima)
            max_frac = max_matches / max_total if max_total > 0 else float('nan')
            min_frac = min_matches / min_total if min_total > 0 else float('nan')

            print(f"[seq_len={seq_len}] Pred local maxima: {[block_labels[i] for i in pred_maxima]}")
            print(f"[seq_len={seq_len}] Act local maxima:   {[block_labels[i] for i in act_maxima]}")
            print(f"[seq_len={seq_len}] Pred local minima:  {[block_labels[i] for i in pred_minima]}")
            print(f"[seq_len={seq_len}] Act local minima:    {[block_labels[i] for i in act_minima]}")
            print(f"[seq_len={seq_len}] Maxima matches (within +/-1): {max_matches}/{max_total} ({max_frac:.2f})")
            print(f"[seq_len={seq_len}] Minima matches (within +/-1): {min_matches}/{min_total} ({min_frac:.2f})")

# Example usage for seq_len_param=8042:
# plot_rq3_analysis_normalized('your_csv.csv', 'your_out_dir', seq_len_param=8042)

if __name__ == "__main__":
    # Define your list of CSV files here
    attn_kernel = "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq3_data/energy_model_results_20250718_040713.csv"

    csv_files = [
        attn_kernel,
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq2_data/experiment_kernels_20250710_043125/energy_model_results_convolution_20250710_044328.csv",
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq2_data/experiment_kernels_20250710_043125/energy_model_results_laplace3d_20250710_043453.csv",
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq2_data/experiment_kernels_20250710_043125/energy_model_results_matMul_20250710_043811.csv",
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq2_data/experiment_kernels_20250710_043125/energy_model_results_reduction_20250710_043853.csv",
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq2_data/experiment_kernels_20250710_043125/energy_model_results_scalarProd_20250710_044122.csv",
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq2_data/experiment_kernels_20250710_043125/energy_model_results_transpose_20250710_044228.csv",
        "/home/anonymous/flipflop/cuda_kernel_energy_empirical/experiments/rq2/rq2_data/experiment_kernels_20250710_043125/energy_model_results_vecAdd_20250710_043620.csv",
    ]
    out_path = "rq3_data/predicted_energy_vs_shape.png"
    actual_out_path = "rq3_data/actual_energy_vs_shape.png"
    actual_seq_len = 8192
    norm_out_dir = "rq3_data/rq3_plots_norm"

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
    # Run plotting for the first file only (for predicted/actual plots)
    plot_predicted_energy_vs_shape(csv_files[0], out_path)
    plot_actual_energy_vs_shape(csv_files[0], actual_out_path, seq_len_param=actual_seq_len)

    # --- Combined normalized extrema-matching analysis ---
    all_results = []
    for csv_file in csv_files:
        print(f"\n=== Analyzing {csv_file} ===")
        if csv_file == attn_kernel:
            def patched_plot_rq3_analysis_normalized(csv_path, out_dir="rq3_plots_norm", seq_len_param=None):
                import copy
                df = pd.read_csv(csv_path)
                if "batch_size" not in df.columns:
                    df["batch_size"] = 1  # fallback
                if "predicted_energy" not in df.columns:
                    if "predicted_power" in df.columns and "predicted_time_ns" in df.columns:
                        df["predicted_energy"] = (
                            df["predicted_power"] * df["predicted_time_ns"] / 1e9
                        ) / (df["batch_size"] * df["seq_len"])
                    else:
                        raise ValueError("CSV must contain 'predicted_energy' or ('predicted_power' and 'predicted_time_ns') columns.")
                if "actual_energy" not in df.columns:
                    if "actual_power" in df.columns and "actual_time_ns" in df.columns:
                        df["actual_energy"] = (
                            df["actual_power"] * df["actual_time_ns"] / 1e9
                        ) / (df["batch_size"] * df["seq_len"])
                    else:
                        raise ValueError("CSV must contain 'actual_energy' or ('actual_power' and 'actual_time_ns') columns.")
                valid_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                df = df[
                    df["block_x"].isin(valid_blocks) &
                    df["block_y"].isin(valid_blocks)
                ].copy()
                df["block_dims"] = df["block_x"].astype(str) + "x" + df["block_y"].astype(str)
                df["threads_per_block"] = df["thread_count"]
                df = df.sort_values(["seq_len", "threads_per_block", "block_x"])
                df["block_dims"] = pd.Categorical(df["block_dims"], ordered=True, categories=df["block_dims"].unique())

                # Normalize predicted and actual energy per token for each sequence length
                df["predicted_energy_norm"] = np.nan
                df["actual_energy_norm"] = np.nan
                for seq_len in df["seq_len"].unique():
                    mask = df["seq_len"] == seq_len
                    pred = df.loc[mask, "predicted_energy"]
                    act = df.loc[mask, "actual_energy"]
                    pred_min, pred_max = pred.min(), pred.max()
                    act_min, act_max = act.min(), act.max()
                    df.loc[mask, "predicted_energy_norm"] = (pred - pred_min) / (pred_max - pred_min + 1e-9)
                    df.loc[mask, "actual_energy_norm"] = (act - act_min) / (act_max - act_min + 1e-9)

                os.makedirs(out_dir, exist_ok=True)
                seq_lens_to_plot = [seq_len_param] if seq_len_param is not None else sorted(df["seq_len"].unique())
                file_results = []
                for seq_len in seq_lens_to_plot:
                    sub = df[df["seq_len"] == seq_len]
                    if sub.empty:
                        print(f"No data for sequence length {seq_len}")
                        continue
                    x = sub["block_dims"]
                    y_pred = sub["predicted_energy_norm"]
                    y_act = sub["actual_energy_norm"]
                    # Plotting omitted for batch mode

                    pred_vals = y_pred.values
                    act_vals = y_act.values
                    block_labels = list(x)

                    def find_local_extrema(arr, mode="max"):
                        extrema = []
                        for i in range(1, len(arr)-1):
                            if mode == "max" and arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                                extrema.append(i)
                            elif mode == "min" and arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                                extrema.append(i)
                        return extrema

                    pred_maxima = find_local_extrema(pred_vals, mode="max")
                    pred_minima = find_local_extrema(pred_vals, mode="min")
                    act_maxima = find_local_extrema(act_vals, mode="max")
                    act_minima = find_local_extrema(act_vals, mode="min")

                    def has_nearby(idx, idx_list, window=1):
                        return any(abs(idx - j) <= window for j in idx_list)

                    max_matches = sum(has_nearby(i, act_maxima) for i in pred_maxima)
                    min_matches = sum(has_nearby(i, act_minima) for i in pred_minima)
                    max_total = len(pred_maxima)
                    min_total = len(pred_minima)
                    max_frac = max_matches / max_total if max_total > 0 else float('nan')
                    min_frac = min_matches / min_total if min_total > 0 else float('nan')

                    print(f"[seq_len={seq_len}] Pred local maxima: {[block_labels[i] for i in pred_maxima]}")
                    print(f"[seq_len={seq_len}] Act local maxima:   {[block_labels[i] for i in act_maxima]}")
                    print(f"[seq_len={seq_len}] Pred local minima:  {[block_labels[i] for i in pred_minima]}")
                    print(f"[seq_len={seq_len}] Act local minima:    {[block_labels[i] for i in act_minima]}")
                    print(f"[seq_len={seq_len}] Maxima matches (within +/-1): {max_matches}/{max_total} ({max_frac:.2f})")
                    print(f"[seq_len={seq_len}] Minima matches (within +/-1): {min_matches}/{min_total} ({min_frac:.2f})")
                    file_results.append({
                        'csv': csv_path,
                        'seq_len': seq_len,
                        'max_matches': max_matches,
                        'max_total': max_total,
                        'max_frac': max_frac,
                        'min_matches': min_matches,
                        'min_total': min_total,
                        'min_frac': min_frac
                    })
                return file_results
            file_results = patched_plot_rq3_analysis_normalized(csv_file, norm_out_dir, seq_len_param=None)
            all_results.extend(file_results)
        else:
            # For non-attn kernels: plot normalized energy per block shape (no seq_len)
            df = pd.read_csv(csv_file)
            if "batch_size" not in df.columns:
                df["batch_size"] = 1  # fallback
            if "predicted_energy" not in df.columns:
                if "predicted_power" in df.columns and "predicted_time_ns" in df.columns:
                    df["predicted_energy"] = (
                        df["predicted_power"] * df["predicted_time_ns"] / 1e9
                    ) / df["batch_size"]
                else:
                    raise ValueError("CSV must contain 'predicted_energy' or ('predicted_power' and 'predicted_time_ns') columns.")
            if "actual_energy" not in df.columns:
                if "actual_power" in df.columns and "actual_time_ns" in df.columns:
                    df["actual_energy"] = (
                        df["actual_power"] * df["actual_time_ns"] / 1e9
                    ) / df["batch_size"]
                else:
                    raise ValueError("CSV must contain 'actual_energy' or ('actual_power' and 'actual_time_ns') columns.")
            df["block_shape"] = df.apply(lambda row: f"{int(row['block_x'])}x{int(row['block_y'])}", axis=1)
            df["thread_count"] = df["block_x"] * df["block_y"]
            df = df.sort_values(["thread_count", "block_x", "block_y"])
            block_shapes_sorted = df.drop_duplicates(["block_shape"])[["thread_count", "block_x", "block_y", "block_shape"]]
            block_shapes_sorted = block_shapes_sorted.sort_values(["thread_count", "block_x", "block_y"])
            block_shape_labels = block_shapes_sorted["block_shape"].tolist()
            # Normalize
            pred = df["predicted_energy"]
            act = df["actual_energy"]
            pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
            act_norm = (act - act.min()) / (act.max() - act.min() + 1e-9)
            x = df["block_shape"]
            plt.figure(figsize=(16, 10))
            plt.plot(x, pred_norm, marker="o", label="Predicted")
            plt.plot(x, act_norm, marker="x", linestyle="--", label="Actual")
            ax = plt.gca()
            ticks = np.arange(len(x))
            ax.set_xticks(ticks)
            new_labels = [label if i % 2 == 0 else "" for i, label in enumerate(x)]
            ax.set_xticklabels(new_labels, rotation=60, fontsize=22)
            plt.yticks(fontsize=18)
            plt.xlabel("Block Shape (X x Y)", fontsize=24)
            plt.ylabel("Normalized Energy per Block", fontsize=24)
            plt.legend(fontsize=20)
            plt.tight_layout()
            out_path = os.path.join(norm_out_dir, f"normalized_energy_per_block_{os.path.basename(csv_file).replace('.csv', '')}.png")
            os.makedirs(norm_out_dir, exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            print(f"Saved normalized plot for {csv_file} to {out_path}")
            print("Extrema-matching analysis skipped for non-attn kernel.")
    # --- Combined summary ---
    if all_results:
        print("\n=== Combined Extrema-Matching Summary ===")
        for r in all_results:
            print(f"CSV: {r['csv']}, seq_len: {r['seq_len']}, Maxima: {r['max_matches']}/{r['max_total']} ({r['max_frac']:.2f}), Minima: {r['min_matches']}/{r['min_total']} ({r['min_frac']:.2f})")
        # Overall averages
        max_fracs = [r['max_frac'] for r in all_results if r['max_total'] > 0]
        min_fracs = [r['min_frac'] for r in all_results if r['min_total'] > 0]
        if max_fracs:
            print(f"Overall average maxima match fraction: {np.mean(max_fracs):.2f}")
        if min_fracs:
            print(f"Overall average minima match fraction: {np.mean(min_fracs):.2f}")