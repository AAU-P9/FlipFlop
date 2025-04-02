#!/bin/bash
# rq3_experiment.sh

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Experiment parameters
KERNEL_FILE="kernels_test/mha.cu"
BATCH_SIZE=4
SEQ_LENS="128,256,512,786,1024,2048,4096,8192"
NHEAD=16
DIM_PER_HEAD=256
CSV_OUT="rq3_data/full_energy_results.csv"
ITERATIONS=5
CALIB_FILE="calibration/calibration_pl250.json"

# Create output directory
mkdir -p "$(dirname "$CSV_OUT")"

# Run the Python script
python energy_model.py \
  --kernel_file "${KERNEL_FILE}" \
  --batch_size  "${BATCH_SIZE}" \
  --seq_lens    "${SEQ_LENS}" \
  --nhead       "${NHEAD}" \
  --dim_per_head "${DIM_PER_HEAD}" \
  --calib       "${CALIB_FILE}" \
  --csv_out     "${CSV_OUT}" \
  --iterations  "${ITERATIONS}"