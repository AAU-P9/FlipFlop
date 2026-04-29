#!/bin/bash
# runner_vecAdd.sh - VecAdd Kernel Energy Analysis

export CUDA_ROOT=/usr/local/cuda-12
export PATH=/usr/local/cuda-12/bin:/usr/bin:$PATH

# Experiment parameters
KERNEL_FILE="kernels/vecAdd.cu"
VECTOR_SIZES="1024,8192,65536,262144,1048576"
CSV_OUT="vecadd_data/energy_results.csv"
ITERATIONS=5
CALIB_FILE="calibration/calibration_pl250.json"

# Create output directory
mkdir -p "$(dirname "$CSV_OUT")"

# Run the Python script
uv run -m energy_model_vecAdd \
  --kernel_file "${KERNEL_FILE}" \
  --vector_sizes "${VECTOR_SIZES}" \
  --calib        "${CALIB_FILE}" \
  --csv_out      "${CSV_OUT}" \
  --iterations   "${ITERATIONS}"
