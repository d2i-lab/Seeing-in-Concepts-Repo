#!/bin/bash
set -e

echo "Starting BDD MaskCLIP tests..."
./test_k4_BDD_250_maskclip.sh
echo "BDD MaskCLIP tests completed."

echo "Starting COCO MaskCLIP tests..."
./test_k4_COCO_250_maskclip.sh
echo "COCO MaskCLIP tests completed."

echo "Starting LVIS MaskCLIP tests..."
./test_k4_LVIS_250_maskclip.sh
echo "LVIS MaskCLIP tests completed."

echo "All test suites completed successfully." 