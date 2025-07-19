#!/bin/bash
# rq4_experiment_llama3.sh
# Example script to run your "energy_model.py" on the llama3.cu kernel

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Example user-set parameters (adjust as needed)
KERNEL_FILE="llama3.cu"
BEAMSIZE=4
N_STEPS=9
NHEAD=16
DIM_PER_HEAD=256
CSV_OUT="rq4_data/llama3_energy_results.csv"
ITERATIONS=5
CALIB_FILE="calibration_pl250.json"

mkdir -p "$(dirname "$CSV_OUT")"


export PYTHONPATH="/home/anonymous/flipflop:$PYTHONPATH"

python llama3_tune.py \
  --kernel_file "${KERNEL_FILE}" \
  --beamsize  "${BEAMSIZE}" \
  --n_steps   "${N_STEPS}" \
  --nhead     "${NHEAD}" \
  --dim_per_head "${DIM_PER_HEAD}" \
  --calib     "${CALIB_FILE}" \
  --csv_out   "${CSV_OUT}" \
  --iterations "${ITERATIONS}"
