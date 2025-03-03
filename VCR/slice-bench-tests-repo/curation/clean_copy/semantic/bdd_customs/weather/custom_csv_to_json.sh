#!/bin/bash

for file in *.csv; do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
        # Run Python command on the file
        python3 ../../coco_cluster_from_csv.py --img_dir /data/users/jie/data-slicing/bdd100k/bdd_select --csv "$file" -n 2500 -e normal_outlier_only -cl -o output/"$file".json --xyxy_to_xywh
    fi
done

