#!/bin/bash
set -e

# PYTHON=/home/jxu680/miniconda3/envs/faiss/bin/python
PYTHON=/data/home/jxu680/miniconda3/envs/faiss/bin/python
EMBED_DIR=/data/jie/slicing/data/COCO/embeds/train2017_siglip/
PK=10
N_QUERIES=1000
INDEX_TYPE="flat"
OUTPUT_DIR="k2_tests_COCO_train2017_SIGLIP_pk(${PK})_n(${N_QUERIES})_idx(${INDEX_TYPE})"
IMAGE_DIR="/data/jie/slicing/data/COCO/train2017"
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Array to store PIDs of background processes
pids=()

# Function to kill all background processes
cleanup() {
    echo "Stopping all processes..."
    for pid in "${pids[@]}"; do
        pkill -P $pid 2>/dev/null || true
        kill -TERM $pid 2>/dev/null || true
    done
    wait
    echo "All processes stopped."
    exit
}

# Set up trap to catch SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Function to run a single test
run_test() {
    echo "Running test for n_segments=$1"
    local n_segments=$1
    local OUTPUT_FILE="${OUTPUT_DIR}/k2_test_n${n_segments}_${PK}.pkl"
    local TIME_OUTPUT="${OUTPUT_DIR}/time_n${n_segments}.txt"
    local PICKLE_FILE="COCO_train2017_SIGLIP_flat_1000_embeds.pkl"
    
    /usr/bin/time -v $PYTHON preprocess/generate_tests.py \
        --embed_dir $EMBED_DIR \
        --out $OUTPUT_FILE \
        --index_type $INDEX_TYPE \
        --n_segments $n_segments \
        --n_queries $N_QUERIES \
        --pk $PK \
        --image_dir $IMAGE_DIR \
        --pickle $PICKLE_FILE \
        2> $TIME_OUTPUT &
    
    local test_pid=$!
    pids+=($test_pid)
    echo "Started test for n_segments=$n_segments (PID: $test_pid)"
}

# Run tests for different n_segments values
for n_segments in 1 2 4 8; do
    run_test $n_segments
done

# Wait for all background processes to finish
wait

echo "All tests completed. Results are in the $OUTPUT_DIR directory."

# Extract and display max memory usage for each test
echo "Memory usage summary:"
for n_segments in 1 2 4 8; do
    TIME_OUTPUT="${OUTPUT_DIR}/time_n${n_segments}.txt"
    MAX_MEM=$(grep "Maximum resident set size" $TIME_OUTPUT | awk '{print $6}')
    echo "n_segments=$n_segments: $MAX_MEM KB"
done
