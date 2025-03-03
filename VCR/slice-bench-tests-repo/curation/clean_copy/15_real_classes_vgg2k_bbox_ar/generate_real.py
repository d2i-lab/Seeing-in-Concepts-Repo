import os
import sys
import json
import argparse

sys.path.append('..')
import coco_cluster_from_csv as cc
import error_settings as es
import img_utils

import pandas as pd

ann = '/home/jxu680/data-slicing-pipeline/mm-post-processing/1-pair-bboxes/vgg_full_improved_80_pair1'
idir = '/data/users/jie/data-slicing/genome/VG_100K'

def get_top_classes(df, n_classes):
    df = df[df['gt_label'] == df['pred_label']]
    counts = df['gt_label'].value_counts()
    return list(counts.index[:n_classes])

def generate_tests(csv_file, img_dir, n_boxes, n_classes, injection, target_dim, 
                   error_type, args_dict):

    df = pd.read_csv(csv_file)
    dim_by_dim = (target_dim, target_dim)

    
    for label_id in get_top_classes(df, n_classes):
        print('Generating for top class: {}'.format(label_id))
        df_slice, img_data = img_utils.get_img_labels(
            df=df.copy(), 
            img_dir=img_dir,
            label=label_id,
            target_dim=dim_by_dim,
            n_imgs=n_boxes
        )
        out_path = '{}_cid_{}_.json'.format(injection, label_id)

        try:
            cc.main(
                df_slice,
                img_data,
                label_id,
                injection,
                out_path,
                target_dim,
                error_type,
                is_clip=True,
                args_dict=args_dict
            )
        except Exception as e:
            print('Error with cid:', label_id, e)
            continue
    
    with open('args.json', 'w') as f:
        json.dump(args_dict, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--json', '-j', type=str, default=ann)
    parser.add_argument('--csv', '-cs', type=str, default=ann)
    parser.add_argument('--img_dir', '-d', type=str, default=idir)
    parser.add_argument('--n_boxes', '-n', type=int, default=2000)
    parser.add_argument('--n_classes', '-c', type=int, default=15)
    parser.add_argument('--injection', '-i', type=str, default='box_aspect_range')
    parser.add_argument('--target_dim', '-t', type=int, default=50, 
                        help='Size of bounding box resized crop used in clustering decisions (n x n)')
    parser.add_argument('--error_type', '-e', type=str, 
                        choices=es.get_error_types().keys(), default='normal_outlier_only',
                        help='Type of error to inject')
    args = parser.parse_args()

    arg_dict = vars(args)

    print('Using annotation file:', args.csv)

    generate_tests(args.csv, args.img_dir, args.n_boxes, args.n_classes, args.injection,
                   args.target_dim, args.error_type,
                   arg_dict)
