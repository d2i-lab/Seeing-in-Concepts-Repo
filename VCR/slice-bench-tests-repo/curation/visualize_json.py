import os
import json
import math
import argparse

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

import img_utils as img_utils

def draw_images(images, rows, cols, out, title=None, save=True):
    '''
    Helper function to draw images in a grid
    '''
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, rows*cols + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(images[i-1])

    axes = fig.get_axes()
    for ax in axes:
        ax.axis('off')
        
    fig.suptitle(title)
    if save:
        plt.savefig(out)
    else:
        plt.show()

def draw_tiles(img_list, tile_dim, title, out, save=True):
    '''
    Calculate the number of rows and columns needed to display
    the images in a grid
    '''
    n_imgs = len(img_list)
    n_size = math.floor(min(tile_dim, math.sqrt(n_imgs)))
    draw_images(img_list, n_size, n_size, out, title=title, save=save)

def main(img_dir, annotations, json_file, tile_dim, out_folder):
    with open(json_file) as f:
        data = json.load(f)

    dataset = COCO(annotations)

    img_name = data['img_name']
    ann_id = data['ann_id']
    target_dim = data['target_dim']
    cluster_label = np.array(data['cluster_label'])
    unique_labels = np.unique(cluster_label)

    img_list = []
    for img_name, ann_id in zip(img_name, ann_id):
        ann = dataset.anns[ann_id]

        # bbox is in xywh format
        bbox = [int(d) for d in ann['bbox']]
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        cropped_img = img_utils.crop_by_bbox_pil(img, bbox)
        cropped_img = img_utils.resize(cropped_img, (target_dim, target_dim))
        img_list.append(cropped_img)

    img_list = np.array(img_list)

    for label in unique_labels:
        idx = np.where(cluster_label == label)[0]
        n_matching = len(idx)
        n_size = math.floor(min(tile_dim, math.sqrt(n_matching)))

        out_name = os.path.join(out_folder, 'label_{}.png'.format(label))
        draw_tiles(img_list[idx], n_size, 'Label: {}'.format(label),
                   out_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', '-i', type=str, default='/data/coco/val2014')
    parser.add_argument('--annotation', '-a', type=str, default='/data/coco/annotations/instances_val2014.json')
    parser.add_argument('--json', '-j', type=str, required=True)
    parser.add_argument('--tile_dim', '-t', type=int, default=10)
    parser.add_argument('--out_folder', '-o', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    main(args.img_dir, args.annotation, args.json, args.tile_dim, args.out_folder)
