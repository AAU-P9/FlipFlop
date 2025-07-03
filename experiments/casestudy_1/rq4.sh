#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Get all built binaries
BINARIES=(runcuda_*)

# print BiNARIES
echo "Binaries found:"
for bin in "${BINARIES[@]}"; do
    echo " - $bin"
done
echo "========================================"
# POWER_LIMITS=(100 250)
POWER_LIMITS=(125 150 175 200 225)

for power_limit in "${POWER_LIMITS[@]}"; do
    echo "========================================"
    echo " Running experiments at ${power_limit}W "
    echo "========================================"
    
    sudo nvidia-smi -pl $power_limit
    
    for bin in "${BINARIES[@]}"; do
        # Extract block size from filename
        block_size="${bin#runcuda_}"
        
        echo "Running ${bin} (block size ${block_size}) at ${power_limit}W..."
        start_time=$(date +%s)
        
        python3 rq4_experiment.py \
            --binary "./${bin}" \
            --power_limit $power_limit \
            --block_size $block_size
        
        # Calculate remaining sleep time
        elapsed=$(( $(date +%s) - start_time ))
        remaining=$(( 60 - elapsed ))
        
        if [ $remaining -gt 0 ]; then
            echo "Sleeping for ${remaining}s before next run..."
            sleep $remaining
        fi
    done
    
    # Extra sleep between power limit changes
    sleep 120
done

# Combine results with headers
echo "Combining results..."
awk 'FNR==1 && NR!=1{next;}{print}' tinystories_*_*W_results.csv > final_results.csv