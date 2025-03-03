#!/bin/bash
set -e

# Configuration
PYTHON=/home/jxu680/miniconda3/envs/faiss/bin/python
CONFIG_FILE="settings/kepler3.yaml"
DATASET="train2017_fixed_maskclip"
TEST_RESULTS_DIR="../k3_tests_results_pk(10)_n(100)_idx(ivf_flat_1)"

# Check if the test results directory exists
if [ ! -d "$TEST_RESULTS_DIR" ]; then
    echo "Error: Test results directory not found: $TEST_RESULTS_DIR"
    exit 1
fi

# Array to store background process PIDs
pids=()

# Function to run a single test
run_test() {
    local test_file=$1
    
    echo "Running test for: $test_file"
    $PYTHON main.py --config $CONFIG_FILE --dataset $DATASET --test_file "$test_file" &
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
        run_test "$test_file"
    fi
done

# Wait for all background processes to finish
wait

echo "All tests completed."
