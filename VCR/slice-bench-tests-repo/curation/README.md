# COCO Curation
This is an attempt to identify/create semi-automated slice-injections over 
the COCO dataset. The hope is that we can use a simple heuristic, like 
"image color" to create partitions/clusters of related images, apply our 
error injection over them, and discover them via concepts.

# Cluster Discovery
```bash
python3 coco_cluster_v1.py --img_dir /data/users/jie/data-slicing/COCO/val2014/ --annotation /data/users/jie/data-slicing/COCO/annotations/instances_val2014.json -n 500 -cid 1 --clustering kmeans_20 -o test.json
```

# Visualize (for debuggin)
```bash
python3 visualize_json.py --json test.json -o test_viz
```

```bash
python3 visualize_json.py  --json birds_10k.json --annotation /data/coco/annotations/instances_val2014.json -o bird_pics
```

# Export

## Our System
```bash
python3 coco_export_v1.py --json kmeans20_cid_1_1k.json --sam /data/users/jie/data-slicing/COCO/sam_jsons/val2014_vit_l -o vcr_kmeans20_cid_1_1k.json -i gt -f one-percent-classic --img /data/users/jie/data-slicing/COCO/val2014/ --embed /data/users/jie/data-slicing/COCO/embeddings/data/coco-2014-val-clip-embeds-532/ -fp /data/users/jie/data-slicing/COCO/embeddings/coco-2014-val-clip-embeds-fast.pkl --padding 50
```

## DOMINO
```bash
python3 domino_eval.py --model clip16 --json test.json --n_slices 10 --padding 50 --out test_results.json
```
