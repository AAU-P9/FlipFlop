#!/bin/bash
# rq2_experiment.sh
# --------------------------------------------------------------------
# Shell script to run the RQ2 experiment for the MHA kernel.
# This script sets up the CUDA environment and launches the Python
# experiment with two different strategies: brute_force and bayesian.
# Each run stores its results in a separate CSV file.
#
# Usage: ./rq2_experiment.sh
# --------------------------------------------------------------------

# Set CUDA environment paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Do not override GPU power limit: we use dynamic power capping via NVMLObserver.

# Experiment parameters
KERNEL_FILE="baselines/mha_kernel.cu"
BATCH_SIZE=4
SEQ_LENS="128,256,512,786,1024,2048,4096,8192"  # Sweeping multiple sequence lengths
NHEAD=16
DIM_PER_HEAD=256
ITERATIONS=10           # Adjust iterations for robust measurements

# Strategies to test
strategies=("brute_force" "bayes_opt")
# strategies=("bayes_opt")

# Create required directories if needed.
mkdir -p "$(dirname "$KERNEL_FILE")"

# Loop over both strategies
for strat in "${strategies[@]}"; do
    echo "[INFO] Launching RQ2 tuning with strategy: $strat"
    CSV_OUT_BASE="rq2_data/mha_tuning_results_${strat}.csv"
    mkdir -p "$(dirname "$CSV_OUT_BASE")"

    python3 rq2.py \
        --kernel_file "$KERNEL_FILE" \
        --batch_size "$BATCH_SIZE" \
        --seq_lens "$SEQ_LENS" \
        --nhead "$NHEAD" \
        --dim_per_head "$DIM_PER_HEAD" \
        --csv_out "$CSV_OUT_BASE" \
        --iterations "$ITERATIONS" \
        --strategy "$strat" \
        --label "rq2_${strat}"
done
