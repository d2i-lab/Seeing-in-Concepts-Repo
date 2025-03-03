#!/bin/bash
json_folders=("clean_copy/semantic/coco_customs_15" "clean_copy/15_real_classes_coco2k" "clean_copy/15_real_classes_coco2k_bbox_ar" "clean_copy/15_real_classes_coco2k_bbox_size")

# Change accordingly
PYTHON_CMD="/data/home/jxu680/miniconda3/envs/rapids-23.12/bin/python3"
SAM_FOLDER="/data/users/jie/data-slicing/COCO/sam_jsons/val2014_vit_l"
IMG_FOLDER="/data/users/jie/data-slicing/COCO/val2014"
EMBED_FOLDER="/data/users/jie/data-slicing/COCO/embeddings/data/coco-2014-val-clip-embeds-532/"
EMBED_PICKLE="/data/users/jie/data-slicing/COCO/embeddings/coco-2014-val-clip-embeds-fast.pkl"
K_CONCEPTS=500
PADDING=50
SUPPORT="10" # String because this argument supports lists (e.g. "10 25")
DUP="0.5"

# Check if all the folders exist
for word in "${json_folders[@]}"; do
    if [ ! -d "${word}" ]; then
        echo "Folder ${word} does not exist. Exiting..."
        exit 1
    fi
done

# Iterate over each folder
for word in "${json_folders[@]}"; do
    # All on
    $PYTHON_CMD coco_export_v1.py --json_folder "$word"\
    --sam $SAM_FOLDER\
    -i gt -f one-percent-classic --img $IMG_FOLDER\
    --embed $EMBED_FOLDER\
    -fp $EMBED_PICKLE\
    --support $SUPPORT --dup_thresh $DUP\
    --padding $PADDING --k_concepts $K_CONCEPTS --allow_absence --allow_box_area --allow_box_aspect &

    pid1=$!
    
    # No absence, no bbox area, no bbox aspect ratio
    $PYTHON_CMD coco_export_v1.py --json_folder "$word"\
    --sam $SAM_FOLDER\
    -i gt -f one-percent-classic --img $IMG_FOLDER\
    --embed $EMBED_FOLDER\
    -fp $EMBED_PICKLE\
    --support $SUPPORT --dup_thresh $DUP\
    --padding $PADDING --k_concepts $K_CONCEPTS &

    pid2=$!

    # No concepts
    $PYTHON_CMD coco_export_v1.py --json_folder "$word"\
    --sam $SAM_FOLDER\
    -i gt -f one-percent-classic --img $IMG_FOLDER\
    --embed $EMBED_FOLDER\
    -fp $EMBED_PICKLE\
    --support $SUPPORT --dup_thresh $DUP\
    --padding $PADDING --k_concepts $K_CONCEPTS --no_concepts &

    pid3=$!

    wait $pid1
    wait $pid2
    wait $pid3
    echo "Next step"
done
