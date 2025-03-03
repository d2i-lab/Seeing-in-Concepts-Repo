# Custom Detections
In case users are interested in running their own detection results over COCO, we recommend the use of MMDetection. Our system is specifically designed to process results from MMDetection and convert it to the proper format for analysis.

## MMDetection Object Detection: Installation and Inference
> [!NOTE]
> Please make sure to satisfy the following Dependencies: (0) MMDetection installation (1) Datasets

We start first with the object detection model itself. Follow the mmdetection installation guide featured on the [mmdetection website](https://mmdetection.readthedocs.io/en/latest/get_started.html). We highly recommend the use of a virtual environment (venv or conda) for this step.

Once the repository is installed, choose a object detection model to evaluate on. For our paper we used MMDetection's R-50-C4 Faster RCNN model found [here](https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn). Download the model weights accordingly. Additionally, verify that the model is compatible with the dataset you plan to evaluate over. 

Modify the configuration file as needed and then run detection inference (we recommend running this in screen or tmux):
```bash
python3 tools/test.py configs/faster_rcnn/{CONFIG_PATH} checkpoints/faster_rcnn_r50_caffe_c4_1x_coco_20220316_150152-3f885b85.pth --out {PICKLE_PATH}.pkl
```
Where `{CONFIG_PATH}` is the model configuration file and `{PICKLE_PATH}` is the inference output directory name. This output pickle file contains metadata information about detection results, including bounding boxes, label confidences, and IoU scores.

## Post-Inference:
We provide several scripts to convert the pickle file into a usable CSV file under the mm-post-processing directory. The first step involves bounding-box pairing and the second step involves finding the interactions of bounding boxes and image segments. These two steps need to be run one after the other in order to obtain a usable CSV file.

### 1. Pairing
Example:
```
1-pair-bboxes$ python3 main.py --bbox_pickle ~/pickles/coco_2014.pkl --pair_mode confidence --out coco_2014_part1
```
* bbox_pickle: The pickle file output from mmdetection inference
* pair_mode: Bounding box pairing method
* out: The output file name


### 2. Interactions
Example:
```
python3 bbox_segments.py --sam /data/sam_jsons/val2014_vit_l/ --bbox coco_2014_part1 --out coco_2014_part2.csv --interact gt
```
* sam: The directory containing all the sam segments for each image
* bbox: The bounding box csv file from part 1, Pairing
* out: The output path
* interact: The bounding box to consider when finding overlaps. If "gt" then finds segments overlap with ground-truth bounding box. If "union", then finds segments overlap with the union of ground-truth and predicted.

Once this is run, the final interaction csv can be used. Modify the configurations in both frontend and backend to reflect this. See [this](/README.md) for more details.
