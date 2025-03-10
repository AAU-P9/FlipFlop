#!/bin/bash
# runner.sh
# Run multiple kernel experiments with the new integrated time and power estimation model.
# 1) Calibrate the GPU once.
# 2) Run for each combination of (kernel, grid, block, data).
#
# The script collects:
#  - Estimated Time (ns)
#  - Actual Median Time (ns)
#  - Difference (%)
#  - Estimated Power (W)
#  - Actual Power (W)
#
# Results are stored in a CSV file.

# Configuration
OUTPUT_DIR="experiment-results-$(date +%Y%m%d-%H%M%S)"
PY_SCRIPT="power_model.py"
KERNEL_DIR="kernels"
RUNS=100  # Number of runs per configuration

# Create output directory
mkdir -p "$OUTPUT_DIR"
CSV_FILE="$OUTPUT_DIR/results.csv"

# 1) Calibrate once:
echo "=== Step 1: Calibrating GPU with time_model.py calibrate ==="
python3 "$PY_SCRIPT" calibrate
if [ $? -ne 0 ]; then
    echo "[ERROR] Calibration failed! Exiting."
    exit 1
fi
echo "Calibration done."

# 2) Prepare CSV output with additional columns for power
echo "Kernel,GridX,BlockX,DataSize,EstimatedTime(ns),ActualTime(ns),Difference(%),EstimatedPower(W),ActualPower(W)" > "$CSV_FILE"

# 3) Define experiment parameters
GRID_SIZES=(256 512 1024)
BLOCK_SIZES=(128 256 512)
DATA_SIZES=(65536 131072 262144 524288 1048576)
KERNELS=("vecAdd.cu" "matMul.cu" "laplace3d.cu")

# 4) Loop over all combinations
for kernel in "${KERNELS[@]}"; do
    for grid_size in "${GRID_SIZES[@]}"; do
        for block_size in "${BLOCK_SIZES[@]}"; do
            for data_size in "${DATA_SIZES[@]}"; do
                echo "--------------------------------------------------"
                echo "Running experiment with:"
                echo "  Kernel:    $kernel"
                echo "  Grid:      $grid_size"
                echo "  Block:     $block_size"
                echo "  DataSize:  $data_size"
                echo "  Runs:      $RUNS"
                echo "--------------------------------------------------"

                output=$(python3 "$PY_SCRIPT" run "$KERNEL_DIR/$kernel" \
                    "$grid_size" \
                    "$block_size" \
                    "$data_size" \
                    "$RUNS" 2>&1)

                if [ $? -ne 0 ]; then
                    echo "[ERROR] Error running $kernel with Grid=$grid_size, Block=$block_size, DataSize=$data_size"
                    echo "$output"
                    continue
                fi

                # Extract the required values from output:
                # Expected output lines:
                #   Estimated Time (ns)  : <value>
                #   Actual Median (ns)   : <value>
                #   Diff (%)             : <value>
                #   Approx Power (W)     : <value>
                #   Actual Power (W)     : <value>
                estimated=$(echo "$output" | grep "Estimated Time (ns)" | awk '{print $5}')
                actual=$(echo "$output" | grep "Actual Median (ns)" | awk '{print $5}')
                diff_pct=$(echo "$output" | grep "Diff (%)" | awk '{print $4}')
                est_power=$(echo "$output" | grep "Approx Power (W)" | awk '{print $5}')
                act_power=$(echo "$output" | grep "Actual Power (W)" | awk '{print $5}')

                if [ -z "$estimated" ] || [ -z "$actual" ] || [ -z "$diff_pct" ] || [ -z "$est_power" ] || [ -z "$act_power" ]; then
                    echo "[WARNING] Could not parse output for $kernel (Grid=$grid_size, Block=$block_size, DataSize=$data_size)."
                    echo "$output"
                    continue
                fi

                echo "$kernel,$grid_size,$block_size,$data_size,$estimated,$actual,$diff_pct,$est_power,$act_power" >> "$CSV_FILE"
                echo "Done -> $kernel, GridX=$grid_size, BlockX=$block_size, DataSize=$data_size"
            done
        done
    done
done

echo "All experiments completed."
echo "Results saved in: $CSV_FILE"
