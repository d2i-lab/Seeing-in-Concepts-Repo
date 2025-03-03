import os
import argparse
import json
from typing import List
from collections import Counter

import numpy as np
import injections.clustering as cl
import injections.bbox as bbox_inj
import img_utils
import error_settings as es

from pycocotools.coco import COCO

DECIMALS = 5

def process_clip_json():
    pass

def main(data: List[img_utils.ImageInfo], cat_id, clustering, out_path, target_dim,
         error_type, is_clip, args_dict):
    '''
    Outputs labels and other metadata to pickle
    '''

    if not 'bbox' in clustering:
        img_list = [d.pixels for d in data]
        img_list = img_utils.preprocess_img_list(img_list)

    print('Applying injection method: {}'.format(clustering))

    if 'box' in clustering:
        labels = bbox_inj.get_box_plugin()[clustering](data)
    else:
        labels = cl.get_clustering_plugin()[clustering](img_list)

    error_types = es.get_error_types()[error_type]

    labels = labels.labels_
    print('Labels:', Counter(labels))

    new_data = {
        'img_name': [],
        'ann_id': [],
        'cluster_label': [],
        'target_dim': target_dim,
        'args_dict': args_dict,
        'cat_id': cat_id,
        'bboxes': [d.bbox for d in data],
        'pred_logits': {}
    }
    for d, cluster_label in zip(data, labels):
        if cluster_label == -1:
            continue
        img_name, ann_id = d.img_name, d.annotation_id
        new_data['img_name'].append(img_name)
        new_data['ann_id'].append(int(ann_id))
        new_data['cluster_label'].append(int(cluster_label))

    unique_labels = set(labels) - {-1}

    # For each level of separation,
    # For each cluster_label as the outlier,
    # Generate a distribution of errors
    for k, v in error_types.items():
        new_data['pred_logits'][k] = {}
        for problem_label in unique_labels:
            outlier_idx = labels == problem_label
            logits = np.zeros(len(labels))
            n_outlier = sum(outlier_idx)
            n_inlier = len(labels) - n_outlier
            inlier_logits = v.inlier(n_inlier, clip=True)
            outler_logits = v.outlier(n_outlier, clip=True)
            logits[outlier_idx] = outler_logits
            logits[~outlier_idx] = inlier_logits
            new_data['pred_logits'][k][int(problem_label)] = [
                round(i, DECIMALS) for i in logits.tolist()
            ]

    with open(out_path, 'w') as f:
        json.dump(new_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', '-i', type=str, default='/data/coco/val2014')
    parser.add_argument('--annotation', '-a', type=str, default='/data/coco/annotations/instances_val2014.json')
    parser.add_argument('--n_imgs', '-n', type=int, default=None)
    parser.add_argument('--cat_id', '-cid', type=int, default=1, 
                        help='Category ID')

    # Main knobs for configuring test generation
    # parser.add_argument('--plugin', '-f', type=str, 
    #                     choices=img_utils.get_plugins().keys(), default=None,
    #                     help='Plugin to apply to image list')
    parser.add_argument('--clustering', '-c', type=str, 
                        choices=cl.get_clustering_plugin().keys(), default='kmeans',
                        help='Clustering algorithm to use')
    parser.add_argument('--target_dim', '-t', type=int, default=50, 
                        help='Size of bounding box resized crop used in clustering decisions (n x n)')
    parser.add_argument('--error_type', '-e', type=str, 
                        choices=es.get_error_types().keys(), default='normal',
                        help='Error type that affects error distribution')
    parser.add_argument('--is_clip', '-cl', action='store_true',
                        help='Indicate that this json file was generated from CLIP labeling.')
    # TODO
    parser.add_argument('--padding', '-p', type=int, default=0, 
                        help='Padding for orignal bounding box ((w+p) x (h+p)). Affects output bbox size--not used for clustering')

    parser.add_argument('--out', '-o', type=str, required=True,
                        help='Output file name')
    args = parser.parse_args()
    args_dict = vars(args)

    if os.path.exists(args.out):
        raise ValueError('Output file already exists')

    if args.is_clip:
        process_clip_json()

    dataset = COCO(args.annotation)

    dim_by_dim = (args.target_dim, args.target_dim)
    skip_img_read = 'bbox' in args.clustering

    data = img_utils.get_imgs_cat_id(dataset, args.img_dir, args.cat_id, 
                               n_imgs=args.n_imgs, target_dim=dim_by_dim, skip_img_read=skip_img_read)

    # if args.plugin:
    #     print('Applying plugin: {}'.format(args.plugin))
    #     img_list = img_utils.get_plugins()[args.focus](img_list)

    main(data, args.cat_id, args.clustering, args.out, args.target_dim, args.error_type, args.is_clip, args_dict)