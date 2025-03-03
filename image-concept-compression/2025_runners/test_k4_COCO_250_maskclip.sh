#!/bin/bash
set -e

# Configuration
PYTHON=/home/jxu680/miniconda3/envs/faiss/bin/python
CONFIG_FILE="settings/kepler4.yaml"
DATASET="train2017_fixed_maskclip"
# TEST_RESULTS_DIR="/home/jxu680/image-concept-compression/data_runs/feb/k4_COCO_250_maskclip"
TEST_RESULTS_DIR="/home/jxu680/image-concept-compression/k4_tests_COCO_MASKCLIP_pk(10)_n(1000)_idx(flat)"
MAIN_SCRIPT="/home/jxu680/image-concept-compression/runners/main.py"

# Check if the test results directory exists
if [ ! -d "$TEST_RESULTS_DIR" ]; then
    echo "Error: Test results directory not found: $TEST_RESULTS_DIR"
    exit 1
fi

# Array to store background process PIDs
pids=()

# Function to run a single test

# run_test_20_64() {
#     local test_file=$1
#     echo "Running test for: $test_file"
#     $PYTHON $MAIN_SCRIPT \
#         --config $CONFIG_FILE \
#         --dataset $DATASET \
#         --test_file "$test_file" \
#         --index_setting probe_20_64 \
#         --build_ivf &
#         # --limit 250 &
#     pids+=($!)
# }


# run_test_twenty_128() {
#     local test_file=$1
    
#     echo "Running test for: $test_file"
#     $PYTHON $MAIN_SCRIPT \
#         --config $CONFIG_FILE \
#         --dataset $DATASET \
#         --test_file "$test_file" \
#         --index_setting probe20_128 &
#         # --build_ivf &
#     # --limit 250 &
#     pids+=($!)
# }

run_test_twenty_256() {
    local test_file=$1
    
    echo "Running test for: $test_file"
    $PYTHON $MAIN_SCRIPT \
        --config $CONFIG_FILE \
        --dataset $DATASET \
        --test_file "$test_file" \
        --index_setting probe20_256 \
        --build_ivf &
        # --limit 250 &
    pids+=($!)
}

# Function to kill all background processes
cleanup() {
    echo "Stopping all tests..."
    for pid in "${pids[@]}"; do
        kill $pid 2>/dev/null
    done
    exit 1
}

# Trap Ctrl+C (SIGINT) and call the cleanup function
trap cleanup SIGINT

# Iterate over all .pkl files in the test results directory
for test_file in "$TEST_RESULTS_DIR"/*.pkl; do
    if [ -f "$test_file" ]; then
        # run_test_20_64 "$test_file"
        # run_test_twenty_128 "$test_file"
        run_test_twenty_256 "$test_file"
    fi
done

# Wait for all background processes to finish
wait

echo "All tests completed."
