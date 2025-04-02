#!/bin/bash
# rq1_experiment.sh
# ----------------------------------------
# Shell script to run RQ1 experiment
#   * enumerates block_size_x,y up to 1024
#   * keeps the GPU at its default power limit
#   * uses either brute_force or bayes_opt strategy
# Usage: ./rq1_experiment.sh

# (Optional) Set up environment
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# If you previously had an nvidia-smi -pl line, comment it out to keep default:
# e.g. # nvidia-smi -pl 300

# Experiment parameters
KERNEL_FILE="baselines/mha_kernel.cu"
BATCH_SIZE=4
SEQ_LENS="128,256,512,786,1024,2048,4096,8192"  # Sweeping multiple sequence lengths
NHEAD=16
DIM_PER_HEAD=256
CSV_OUT="rq1_data/mha_tuning_results.csv"
ITERATIONS=5           # Can increase if you need higher accuracy
STRATEGY="brute_force" # or "bayes_opt" if you prefer

# Create directory if needed
mkdir -p "$(dirname "$CSV_OUT")"

# Run the Python script
python3 rq1.py \
  --kernel_file "${KERNEL_FILE}" \
  --batch_size  "${BATCH_SIZE}" \
  --seq_lens    "${SEQ_LENS}" \
  --nhead       "${NHEAD}" \
  --dim_per_head "${DIM_PER_HEAD}" \
  --csv_out     "${CSV_OUT}" \
  --iterations  "${ITERATIONS}" \
  --strategy    "${STRATEGY}" \
  --label       "rq1_adaptive_expt"
