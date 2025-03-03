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

    /data/home/jxu680/miniconda3/envs/rapids-23.12/bin/python3 coco_export_v1.py\
    --json_folder "${word}"\
    --img /data/users/jie/data-slicing/COCO/val2014\
    --embed /data/users/jie/data-slicing/COCO/embeddings/coco-2014-dino-vitb\
    -fp /data/users/jie/data-slicing/COCO/embeddings/coco_val_dino_b_fast.pkl\
    --sam /data/users/jie/data-slicing/COCO/sam_jsons/val2014_vit_l\
    -i gt -f one-percent-classic\
    --padding 50 --k_concepts 500\
    --allow_absence --allow_box_area --allow_box_aspect --force &

    # pid1=$!

    /data/home/jxu680/miniconda3/envs/rapids-23.12/bin/python3 coco_export_v1.py\
    --json_folder "${word}"\
    --img /data/users/jie/data-slicing/COCO/val2014\
    --embed /data/users/jie/data-slicing/COCO/embeddings/coco-2014-dino-vitb\
    -fp /data/users/jie/data-slicing/COCO/embeddings/coco_val_dino_b_fast.pkl\
    --sam /data/users/jie/data-slicing/COCO/sam_jsons/val2014_vit_l\
    -i gt -f one-percent-classic\
    --padding 50 --k_concepts 500 --force &

    # pid2=$!

    # /data/home/jxu680/miniconda3/envs/rapids-23.12/bin/python3 coco_export_v1.py\
    # --json_folder "${word}"\
    # --img /data/users/jie/data-slicing/COCO/val2014\
    # --embed /data/users/jie/data-slicing/COCO/embeddings/data/coco-2014-dino-vitb\
    # -fp /data/users/jie/data-slicing/COCO/embeddings/coco_val_dino_b_fast.pkl\
    # -fp /data/users/jie/data-slicing/COCO/embeddings/coco-2014-val-clip-embeds-fast.pkl\
    # --sam /data/users/jie/data-slicing/COCO/sam_jsons/val2014_vit_l\
    # -i gt -f one-percent-classic\
    # --padding 50 --k_concepts 500 --no_concepts --force&
    
    # wait $pid1
    # wait $pid2
    # wait $pid3

    ((counter++))

    # Check if the maximum number of background processes has been reached
    if (( counter % max_processes == 0 )); then
        # Wait for the background processes to finish before starting the next batch
        echo "Waiting for background processes to finish..."
        wait
        echo "Background processes finished.. Moving on to the next batch."
    fi
done
