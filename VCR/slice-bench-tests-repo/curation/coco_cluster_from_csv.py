import os
import argparse
import json
from typing import List
from collections import Counter

import pandas as pd
import numpy as np
import injections.clustering as cl
import injections.bbox as bbox_inj
import img_utils
import error_settings as es

DECIMALS = 5

def process_clip_json(df, img_dir, target_dim, n_imgs, error_type, xyxy_to_xywh, out_path):
    print('CLIP MODE')
    df, img_data = img_utils.get_img_clip_labels(df, img_dir, target_dim, n_imgs, read_imgs=False)
    if 'label' in df.columns:
        labels = list(df['label'])
    else:
        labels = list(df['query'])

    if error_type != 'normal_outlier_only':
        raise Exception('Probably shouldnt do this')

    if xyxy_to_xywh:
        print('Converting xyxy to xywh')
        for d in img_data:
            d.bbox = img_utils.bbox_xyxy_to_xywh(d.bbox)

    error_types = es.get_error_types()[error_type]
    new_data = {
        'img_name': [],
        'ann_id': [],
        'cluster_label': [],
        'target_dim': target_dim,
        'args_dict': args_dict,
        # 'cat_id': cat_id,
        'label_id': 'CUSTOM',
        'bboxes': [d.bbox for d in img_data],
        'pred_logits': {}
    }

    for d, cluster_label in zip(img_data, labels):
        img_name, ann_id = d.img_name, d.annotation_id
        new_data['img_name'].append(img_name)
        new_data['ann_id'].append(int(ann_id))
        new_data['cluster_label'].append((cluster_label))

    unique_labels = set(labels) - {-1}
    original_logits = df['iou']

    labels = np.array(labels)

    # For each level of separation,
    # For each cluster_label as the outlier,
    # Generate a distribution of errors
    for k, v in error_types.items():
        new_data['pred_logits'][k] = {}
        for problem_label in unique_labels:
            outlier_idx = labels == problem_label
            print(sum(outlier_idx), 'is count match')
            # logits = np.zeros(len(labels))
            logits = np.array(original_logits).copy()
            n_outlier = sum(outlier_idx)
            # n_inlier = len(labels) - n_outlier
            # inlier_logits = v.inlier(n_inlier, clip=True)
            outler_logits = v.outlier(n_outlier, clip=True)
            logits[outlier_idx] = outler_logits
            # logits[~outlier_idx] = inlier_logits
            new_data['pred_logits'][k][problem_label] = [
                round(i, DECIMALS) for i in logits.tolist()
            ]

    with open(out_path, 'w') as f:
        json.dump(new_data, f)


def main(df, data: List[img_utils.ImageInfo], label_id, clustering, out_path, target_dim,
         error_type, is_clip, args_dict):
    '''
    Outputs labels and other metadata to pickle
    '''

    print('Applying clustering: {}'.format(clustering))

    if 'box' in clustering:
        labels = np.array(bbox_inj.get_box_plugin()[clustering](data))
    else:
        img_list = [d.pixels for d in data]
        img_list = img_utils.preprocess_img_list(img_list)

        labels = cl.get_clustering_plugin()[clustering](img_list)
        labels = labels.labels_

    error_types = es.get_error_types()[error_type]
    # df['cluster_label'] = labels.labels_

    print('Labels:', Counter(labels))

    new_data = {
        'img_name': [],
        'ann_id': [],
        'cluster_label': [],
        'target_dim': target_dim,
        'args_dict': args_dict,
        # 'cat_id': cat_id,
        'label_id': label_id,
        # 'bboxes': [d.bbox for d in data],
        "bboxes": [],
        'pred_logits': {}
    }

    good_idx = labels != -1
    for d, cluster_label in zip(data, labels):
        if cluster_label == -1:
            continue
        img_name, ann_id = d.img_name, d.annotation_id
        new_data['img_name'].append(img_name)
        new_data['ann_id'].append(int(ann_id))
        new_data['cluster_label'].append(int(cluster_label))
        new_data['bboxes'].append(d.bbox)

    unique_labels = set(labels) - {-1}
    original_logits = df['iou']

    if sum(good_idx) != len(original_logits):
        print('[Warning!]: mismatched length of labels and logits', sum(good_idx), len(original_logits))
        labels = labels[good_idx]
        original_logits = original_logits[good_idx]
        print('Adjusted length of labels and logits', len(labels), len(original_logits))

    # For each level of separation,
    # For each cluster_label as the outlier,
    # Generate a distribution of errors
    for k, v in error_types.items():
        new_data['pred_logits'][k] = {}
        for problem_label in unique_labels:
            # print(labels, problem_label)
            outlier_idx = labels == problem_label
            # print(sum(outlier_idx), 'is count match')
            # logits = np.zeros(len(labels))
            logits = np.array(original_logits).copy()
            n_outlier = sum(outlier_idx)
            # n_inlier = len(labels) - n_outlier
            # inlier_logits = v.inlier(n_inlier, clip=True)
            outler_logits = v.outlier(n_outlier, clip=True)
            logits[outlier_idx] = outler_logits
            # logits[~outlier_idx] = inlier_logits
            new_data['pred_logits'][k][int(problem_label)] = [
                round(i, DECIMALS) for i in logits.tolist()
            ]

    # print(new_data)
    with open(out_path, 'w') as f:
        json.dump(new_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', '-i', type=str, default='/data/coco/val2014')
    parser.add_argument('--csv', '-a', type=str, default='/data/coco/annotations/instances_val2014.json')
    parser.add_argument('--n_imgs', '-n', type=int, default=None)
    parser.add_argument('--label_id', '-lid', type=int, default=1, 
                        help='Label id (starts from 0)--slightly different from category id')

    # Main knobs for configuring test generation
    parser.add_argument('--clustering', '-c', type=str, 
                        choices=cl.get_clustering_plugin().keys(), default='kmeans_20',
                        help='Clustering algorithm to use')
    parser.add_argument('--target_dim', '-t', type=int, default=50, 
                        help='Size of bounding box resized crop used in clustering decisions (n x n)')
    parser.add_argument('--error_type', '-e', type=str, 
                        choices=es.get_error_types().keys(), default='normal_outlier_only',
                        help='Error type that affects error distribution')
    parser.add_argument('--is_clip', '-cl', action='store_true',
                        help='Indicate that this json file was generated from CLIP labeling.')
    parser.add_argument('--xyxy_to_xywh', action='store_true',
                        help='Convert xyxy to xywh (for bbox format)')

    parser.add_argument('--out', '-o', type=str, required=True,
                        help='Output file name')
    args = parser.parse_args()
    args_dict = vars(args)

    if os.path.exists(args.out):
        raise ValueError('Output file already exists')

    dim_by_dim = (args.target_dim, args.target_dim)
    if args.is_clip:
        df = pd.read_csv(args.csv)
        process_clip_json(df, args.img_dir, dim_by_dim, args.n_imgs, args.error_type, args.xyxy_to_xywh, args.out)
        exit(0)


    skip_img_read = 'bbox' in args.clustering
    if skip_img_read:
        print('We skip those')

    df = pd.read_csv(args.csv)
    df, img_data = img_utils.get_img_labels(
        df=df, 
        img_dir=args.img_dir,
        label=args.label_id,
        target_dim=dim_by_dim,
        n_imgs=args.n_imgs,
        skip_img_read=skip_img_read
    )


    main(df, img_data, args.label_id, args.clustering, args.out, args.target_dim, args.error_type, args.is_clip, args_dict)