#!/bin/bash
# calibration.sh
# --------------------------------------------------------------------
# This script runs the GPU calibration at different power limits.
# For each specified power limit, it sets the GPU power limit using nvidia-smi,
# then runs calibration.py and saves the output in a file named:
# calibration/calibration_pl<current_power_limit>.json
#
# Usage: ./calibration.sh
# --------------------------------------------------------------------

# Create calibration folder if it doesn't exist.
mkdir -p calibration

# Define the array of desired power limits in Watts.
power_limits=(100 140 175 210 250)

# Loop over each power limit.
for pl in "${power_limits[@]}"; do
    echo "-------------------------------------------------"
    echo "[INFO] Setting GPU power limit to ${pl}W..."
    sudo nvidia-smi -pl ${pl}
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to set GPU power limit to ${pl}W. Exiting."
        exit 1
    fi

    # Wait a few seconds for the power limit change to take effect.
    sleep 5

    # Override the calibration file location via an environment variable.
    CALIBRATION_FILE="calibration/calibration_pl${pl}.json"
    echo "[INFO] Running calibration at ${pl}W..."
    python3 calibration.py --output ${CALIBRATION_FILE}
    if [ $? -ne 0 ]; then
        echo "[ERROR] Calibration failed at ${pl}W. Exiting."
        exit 1
    fi
    echo "[INFO] Calibration complete for ${pl}W. Results saved to ${CALIBRATION_FILE}"
done

echo "-------------------------------------------------"
echo "[INFO] All calibrations completed successfully."
