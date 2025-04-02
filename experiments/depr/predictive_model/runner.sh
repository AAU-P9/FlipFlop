#!/bin/bash
# runner.sh
# This runner executes kernel experiments using the integrated time and power models.
# It automatically loads a configuration file (if available) from the CONFIG_DIR.
# CSV columns:
# Kernel,BlockX,BlockY,ThreadCount,GridX,GridY,DataSize,EstTime(ns),ActTime(ns),
# DiffTime(%),WarpsPerSM,PredictedPower(W),ActualPower(W),MemCoal,MemUncoal,MemPartial,
# LocalInsts,SharedInsts,SynchInsts,FpInsts,IntInsts,SfuInsts,AluInsts,TotalInsts,RegsPerThread,SharedMemBytes
#
# Kill all child processes on exit
trap "kill 0" EXIT

# Directories and parameters
OUTPUT_DIR="experiment-results"
PREDICT_SCRIPT="energy_model.py"
KERNEL_DIR="kernels"
CONFIG_DIR="kernel_launch_configs"
RUNS=20               # Number of runs per configuration
SLEEP_TIME=2          # Seconds to sleep between experiments

# Total threads per block to test
THREAD_COUNTS=(32 64 128 256 512 1024)
# Grid sizes (using 1D grid for simplicity)
GRID_SIZES=(256)
# All kernel source files in KERNEL_DIR
KERNELS=($(ls ${KERNEL_DIR}/*.cu))

# Create output directory and CSV file
mkdir -p "$OUTPUT_DIR"
CSV_FILE="$OUTPUT_DIR/results-$(date +%Y%m%d-%H%M%S).csv"

# Write CSV header.
echo "Kernel,BlockX,BlockY,ThreadCount,GridX,GridY,DataSize,EstTime(ns),ActTime(ns),DiffTime(%),WarpsPerSM,PredictedPower(W),ActualPower(W),MemCoal,MemUncoal,MemPartial,LocalInsts,SharedInsts,SynchInsts,FpInsts,IntInsts,SfuInsts,AluInsts,TotalInsts,RegsPerThread,SharedMemBytes" > "$CSV_FILE"

# Function to generate factor pairs for a given total T
get_factor_pairs() {
    local T=$1
    python3 <<EOF
T = $T
pairs = []
for bx in range(1, T+1):
    if T % bx == 0:
        by = T // bx
        pairs.append(f"{bx} {by}")
for pair in pairs:
    print(pair)
EOF
}

# Loop over each kernel.
for kernel in "${KERNELS[@]}"; do
    base=$(basename "$kernel")
    # Expected config file: config_<kernelNameWithoutExtension>.json
    config_file="$CONFIG_DIR/config_${base%.*}.json"
    if [ ! -f "$config_file" ]; then
        echo "[WARNING] No config file found for $base, using default."
        config_file=""
    fi

    for grid_size in "${GRID_SIZES[@]}"; do
        gx=$grid_size
        gy=1
        for thread_count in "${THREAD_COUNTS[@]}"; do
            mapfile -t pairs < <(get_factor_pairs $thread_count)
            for pair in "${pairs[@]}"; do
                bx=$(echo "$pair" | awk '{print $1}')
                by=$(echo "$pair" | awk '{print $2}')
                total_threads=$((bx * by))
                for data_size in 131072; do  # You can add more data sizes if needed.
                    echo "--------------------------------------------------"
                    echo "Kernel:       $base"
                    echo "Config:       $config_file"
                    echo "Block shape:  ($bx, $by) => total threads = $total_threads"
                    echo "Grid:         ($gx, $gy)"
                    echo "DataSize:     $data_size"
                    echo "Runs:         $RUNS"
                    echo "--------------------------------------------------"
                    
                    output=$(python3 "$PREDICT_SCRIPT" "$kernel" "$gx" "$gy" "$bx" "$by" "$data_size" --runs "$RUNS" --config "$config_file" 2>&1)
                    if [ $? -ne 0 ]; then
                        echo "[ERROR] energy_model.py failed for $base with block=($bx,$by) grid=($gx,$gy) data=$data_size"
                        echo "$output"
                        continue
                    fi
                    
                    est_time=$(echo "$output" | grep -i "Estimated Time (ns)" | awk -F= '{print $2}' | xargs)
                    act_time=$(echo "$output" | grep -i "Actual Time" | awk -F= '{print $2}' | head -n1 | xargs)
                    diff_time=$(echo "$output" | grep -i "diff (%)" | awk -F= '{print $2}' | xargs)
                    warps=$(echo "$output" | grep -i "WarpsPerSM" | awk -F= '{print $2}' | xargs)
                    pred_power=$(echo "$output" | grep -i "Predicted Power (W)" | awk -F= '{print $2}' | xargs)
                    act_power=$(echo "$output" | grep -i "Actual Power (W)" | awk -F= '{print $2}' | head -n1 | xargs)
                    
                    mem_coal=$(echo "$output" | grep -i "MemCoal=" | awk -F= '{print $2}' | xargs)
                    mem_uncoal=$(echo "$output" | grep -i "MemUncoal=" | awk -F= '{print $2}' | xargs)
                    mem_partial=$(echo "$output" | grep -i "MemPartial=" | awk -F= '{print $2}' | xargs)
                    local_insts=$(echo "$output" | grep -i "LocalInsts=" | awk -F= '{print $2}' | xargs)
                    shared_insts=$(echo "$output" | grep -i "SharedInsts=" | awk -F= '{print $2}' | xargs)
                    synch_insts=$(echo "$output" | grep -i "SynchInsts=" | awk -F= '{print $2}' | xargs)
                    fp_insts=$(echo "$output" | grep -i "FpInsts=" | awk -F= '{print $2}' | xargs)
                    int_insts=$(echo "$output" | grep -i "IntInsts=" | awk -F= '{print $2}' | xargs)
                    sfu_insts=$(echo "$output" | grep -i "SfuInsts=" | awk -F= '{print $2}' | xargs)
                    alu_insts=$(echo "$output" | grep -i "AluInsts=" | awk -F= '{print $2}' | xargs)
                    total_insts=$(echo "$output" | grep -i "TotalInsts=" | awk -F= '{print $2}' | xargs)
                    regs_pt=$(echo "$output" | grep -i "RegsPerThread=" | awk -F= '{print $2}' | xargs)
                    shared_mem_bytes=$(echo "$output" | grep -i "SharedMemBytes=" | awk -F= '{print $2}' | xargs)
                    grid_x=$(echo "$output" | grep -i "GridX=" | awk -F= '{print $2}' | xargs)
                    grid_y=$(echo "$output" | grep -i "GridY=" | awk -F= '{print $2}' | xargs)
                    block_x=$(echo "$output" | grep -i "BlockX=" | awk -F= '{print $2}' | xargs)
                    block_y=$(echo "$output" | grep -i "BlockY=" | awk -F= '{print $2}' | xargs)

                    echo "$base,$block_x,$block_y,$total_threads,$grid_x,$grid_y,$data_size,$est_time,$act_time,$diff_time,$warps,$pred_power,$act_power,$mem_coal,$mem_uncoal,$mem_partial,$local_insts,$shared_insts,$synch_insts,$fp_insts,$int_insts,$sfu_insts,$alu_insts,$total_insts,$regs_pt,$shared_mem_bytes" >> "$CSV_FILE"
                    echo "Experiment completed: Kernel=$base, block=($bx,$by), grid=($gx,$gy), DataSize=$data_size"
                    sleep $SLEEP_TIME
                done
            done
        done
    done
done

echo "All experiments completed."
echo "Results saved in: $CSV_FILE"
