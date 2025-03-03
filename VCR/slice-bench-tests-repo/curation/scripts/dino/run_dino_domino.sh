#!/bin/bash

# List of words
json_folders=("clean_copy/semantic/coco_customs_15" "clean_copy/15_real_classes_coco2k" "clean_copy/15_real_classes_coco2k_bbox_ar" "clean_copy/15_real_classes_coco2k_bbox_size")
counter=0
max_processes=2

# Check if all the folders exist
for word in "${json_folders[@]}"; do
    if [ ! -d "${word}" ]; then
        echo "Folder ${word} does not exist. Exiting..."
        exit 1
    fi
done


# Iterate over each word
for word in "${json_folders[@]}"; do
    modulo=$((counter % 2))
    cuda_device="cuda:${modulo}"

    /data/home/jxu680/miniconda3/envs/domino/bin/python3 domino_eval.py --json_folder "${word}" --model dino_b --padding 50 --device "${cuda_device}" --force &
    ((counter++))

    # Check if the maximum number of background processes has been reached
    if (( counter % max_processes == 0 )); then
        # Wait for the background processes to finish before starting the next batch
        echo "Waiting for background processes to finish..."
        wait
        echo "Background processes finished.. Moving on to the next batch."
    fi
done
