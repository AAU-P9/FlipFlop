#!/bin/bash
# rq3_enhanced.sh
# --------------------------------------------------------------------
# Enhanced RQ3 experiment with comprehensive data collection
# --------------------------------------------------------------------

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Experiment parameters
KERNEL_FILE="baselines/mha_kernel.cu"
BATCH_SIZE=4
SEQ_LENS="128,256,512,786,1024,2048,4096,8192"
NHEAD=16
DIM_PER_HEAD=256
ITERATIONS=5  # Align with paper methodology
STRATEGY="brute_force"
LABEL="rq3_enhanced"
CSV_OUT="rq3_data/mha_comprehensive_results.csv"

# Create required directories
mkdir -p "$(dirname "$KERNEL_FILE")"
mkdir -p "$(dirname "$CSV_OUT")"

echo "[RQ3] Starting enhanced data collection for sequence lengths: $SEQ_LENS"
echo "[RQ3] Output: $CSV_OUT"

python3 rq3.py \
    --kernel_file "$KERNEL_FILE" \
    --batch_size "$BATCH_SIZE" \
    --seq_lens "$SEQ_LENS" \
    --nhead "$NHEAD" \
    --dim_per_head "$DIM_PER_HEAD" \
    --csv_out "$CSV_OUT" \
    --iterations "$ITERATIONS" \
    --strategy "$STRATEGY" \
    --label "$LABEL"

echo "[RQ3] Experiment complete. Data saved to $CSV_OUT"