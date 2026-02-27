#!/bin/bash
# Auto-select an idle GPU with sufficient free memory.
# Usage: source base_select_gpu.sh
# Sets CUDA_VISIBLE_DEVICES to the best available GPU index.

MIN_FREE_MB=${MIN_FREE_MB:-20480}   # default: 20 GB (override before sourcing)

print_gpu_memory() {
    echo "Current GPU memory usage:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits
}

find_available_gpu() {
    print_gpu_memory
    AVAILABLE_GPU=$(nvidia-smi --query-gpu=index,memory.free \
        --format=csv,noheader,nounits | \
        awk -v min_mem="$MIN_FREE_MB" '$2 >= min_mem {print $1, $2}' | \
        sort -k2 -rn | head -n1 | awk '{print $1}')

    if [ -z "$AVAILABLE_GPU" ]; then
        echo "[base_select_gpu] No GPU with >= ${MIN_FREE_MB} MB free. Exiting."
        exit 1
    fi
    echo "[base_select_gpu] Selected GPU $AVAILABLE_GPU (>= ${MIN_FREE_MB} MB free)."
}

find_available_gpu
export CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU
