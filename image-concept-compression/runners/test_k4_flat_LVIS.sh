#!/bin/bash
set -e

# Configuration
PYTHON=/home/jxu680/miniconda3/envs/faiss/bin/python
CONFIG_FILE="settings/kepler4.yaml"
DATASET="lvis_maskclip"
TEST_RESULTS_DIR="../k4_tests_LVIS_MASKCLIP_pk(10)_n(100)_idx(flat)"

# Check if the test results directory exists
if [ ! -d "$TEST_RESULTS_DIR" ]; then
    echo "Error: Test results directory not found: $TEST_RESULTS_DIR"
    exit 1
fi

# Array to store background process PIDs
pids=()

# Function to run a single test

run_test_twenty() {
    local test_file=$1
    
    echo "Running test for: $test_file"
    $PYTHON main.py \
        --config $CONFIG_FILE \
        --dataset $DATASET \
        --test_file "$test_file" \
        --index_setting thousand_twenty_probe &
    pids+=($!)
}
run_test_ten() {
    local test_file=$1
    
    echo "Running test for: $test_file"
    $PYTHON main.py \
        --config $CONFIG_FILE \
        --dataset $DATASET \
        --test_file "$test_file" \
        --index_setting thousand_ten_probe &
    pids+=($!)
}

run_test_five() {
    local test_file=$1
    
    echo "Running test for: $test_file"
    $PYTHON main.py \
        --config $CONFIG_FILE \
        --dataset $DATASET \
        --test_file "$test_file" \
        --index_setting thousand_five_probe \
        --build_ivf &
    pids+=($!)
}

run_test_one() {
    local test_file=$1
    
    echo "Running test for: $test_file"
    $PYTHON main.py \
        --config $CONFIG_FILE \
        --dataset $DATASET \
        --test_file "$test_file" \
        --index_setting thousand_one_probe &
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
        run_test_twenty "$test_file"
        run_test_ten "$test_file"
        run_test_five "$test_file"
        run_test_one "$test_file"
    fi
done

# Wait for all background processes to finish
wait

echo "All tests completed."
