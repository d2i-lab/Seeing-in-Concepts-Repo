import json
import subprocess
import datetime
import pickle
import os
import argparse

import pair_confidence
import pair_iou

MAX_DETECTIONS = 100
MIN_IOU = 0.1
FP_IOU_THRESH = 0.5
MIN_CONFIDENCE = 0.5
pair_choices = ['confidence', 'maximize-iou']

def get_git_revision_hash() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode('ascii').strip()

def add_header(args, in_file):
    config = vars(args)
    config['time'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    config['git-hash'] = get_git_revision_hash()
    config_str = json.dumps(config)
    #TODO: Prepend to in_file

def main(data, pairing: str, out_path: str, max_detections:int,
         min_iou: int, min_confidence: int, fp_iou_thresh: int,
         no_union: bool):
    if pairing == 'confidence':
        pair_class = pair_confidence.PairConfidence(
            pickle_data=data,
            max_detections=max_detections,
            min_iou=min_iou,
            min_confidence=min_confidence,
            fp_iou_thresh=fp_iou_thresh,
            no_union=no_union,
        )
        df_out = pair_class.pair()
        df_out.to_csv(out_path)
    elif pairing == 'maximize-iou':
        pair_class = pair_iou.PairIOU(
            pickle_data=data,
            min_iou=min_iou,
            min_confidence=min_confidence,
            fp_iou_thresh=fp_iou_thresh,
        )
        df_out = pair_class.pair()
        df_out.to_csv(out_path)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bbox_pickle', type=str, required=True,
                        help='Path to mmdetection pickle file')
    parser.add_argument('--pair_mode', type=str, choices=pair_choices, required=True,
                        help='Bounding box pairing method. Either greedy or hungarian method.')
    parser.add_argument('--out', type=str, required=True,
                        help='Output path')
    parser.add_argument('--max_detections', type=int, default=MAX_DETECTIONS)
    parser.add_argument('--min_iou', type=int, default=MIN_IOU)
    parser.add_argument('--min_confidence', type=int, default=MIN_CONFIDENCE)
    parser.add_argument('--fp_iou_thresh', type=int, default=FP_IOU_THRESH)
    # parser.add_argument('--no_union', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.out):
        print('Out path: {} already exists'.format(args.out))
        exit(1)

    if not os.path.exists(args.bbox_pickle):
        print('Path to pickle file: {} does not exist'.format(args.bbox_pickle))
        exit(1) 

    with open(args.bbox_pickle, 'rb') as f:
        data = pickle.load(f)        

    main(data, args.pair_mode, args.out, args.max_detections, args.min_iou,
         args.min_confidence, args.fp_iou_thresh, True)
        #  args.min_confidence, args.fp_iou_thresh, args.no_union)

    