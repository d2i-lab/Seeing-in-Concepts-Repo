import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

@dataclass
class ImageInfo:
    img_name: str
    annotation_id: int
    pixels: list
    bbox: list # [x, y, w, h]

    pred_bbox: Optional[list] = None

def crop_by_bbox(img, bbox):
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def crop_by_bbox_pil(img, bbox):
    x, y, w, h = bbox
    return img.crop((x, y, x+w, y+h))

def crop_by_bbox_xyxy(img, bbox):
    x1, y1, x2, y2 = bbox
    return img.crop((x1, y1, x2, y2))

def resize(img, target_dim):
    img = img.resize(target_dim)
    return np.array(img)

def bbox_xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2-x1, y2-y1]

def parse_bbox_str(bbox_str):
   return np.fromstring(bbox_str[1:-1], dtype=float, sep=',') 

# TODO: Add option to give padding to bounding box crop
def get_imgs_cat_id(dataset, img_dir, cat_id, target_dim=(50,50), n_imgs=None,
                    skip_img_read=False):
    '''
    Given dataset and category id, return list of tuples of 
    (img_name, ann_id, img_array)
    '''
    anns = dataset.getAnnIds(catIds=[cat_id])[:n_imgs]
    img_list = []

    for a_id in anns:
        
        ann = dataset.anns[a_id]
        if 'bbox' not in ann:
            continue

        bbox = [int(d) for d in ann['bbox']]

        img_name = dataset.imgs[ann['image_id']]['file_name']
        cropped_img = np.array([])
        if not skip_img_read:
            # img = Image.open('/data/coco/val2014/{}'.format(img_name)).convert('RGB')
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            cropped_img = crop_by_bbox_pil(img, bbox)
            cropped_img = resize(cropped_img, target_dim)

            # img_tuple = (img_name, a_id, cropped_img.flatten())
            # img_list.append(img_tuple)
        img_data = ImageInfo(img_name, a_id, cropped_img.flatten(), bbox)
        img_list.append(img_data)

        if n_imgs and len(img_list) >= n_imgs:
            break
    
    return img_list

def get_img_labels(df, img_dir, label, target_dim=(50,50), n_imgs=None):
    '''
    Given dataset and category id, return list of tuples of 
    (img_name, ann_id, img_array)
    '''
    img_list = []

    # Make sure that labels match
    df = df[df['gt_label'] == label]
    df = df[df['gt_label'] == df['pred_label']]
    df = df.sample(frac=1, random_state=42)

    for idx, row in df.iterrows():
        
        img_name = row['img_path']
        gt_bbox = parse_bbox_str(row['gt_bbox'])
        pred_bbox = parse_bbox_str(row['pred_bbox'])

        # We probably don't need to do unions if there is sufficient padding
        # pred_bbox = parse_bbox_str(row['pred_bbox'])
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')


        cropped_img = crop_by_bbox_xyxy(img, gt_bbox) # bbox is in xyxy format
        cropped_img = resize(cropped_img, target_dim)
        # cropped_img.save(f'./{label}_{idx}.png')
        # cropped_padding = crop_by_bbox_xyxy(img, do_padding_xyxy(img.size[::-1], bbox, 50))
        # cropped_padding.save(f'./{label}_{idx}_padding.png')
        # raise Exception('saved')

        # Feed bbox in xywh format
        bbox_xywh = bbox_xyxy_to_xywh(gt_bbox)
        pred_bbox_xywh = bbox_xyxy_to_xywh(pred_bbox)

        # The annotation id is replaced with index
        img_data = ImageInfo(img_name, idx, cropped_img.flatten(), bbox_xywh,
                             pred_bbox=pred_bbox_xywh)
        img_list.append(img_data)

        if n_imgs and len(img_list) >= n_imgs:
            break

    print(len(img_list), 'is len of images')
    
    return df[:len(img_list)], img_list

def get_img_clip_labels(df, img_dir, target_dim=(50,50), n_imgs=None, read_imgs=True):
    '''
    Given dataset and category id, return list of tuples of 
    (img_name, ann_id, img_array)
    '''
    img_list = []

    # Make sure that labels match
    df = df.sample(frac=1, random_state=42)
    df = df[df['iou'] >= 0]
    for idx, row in df.iterrows():
        
        img_name = row['img_path']
        gt_bbox = parse_bbox_str(row['bbox'])
        # gt_bbox = parse_bbox_str(row['gt_bbox'])
        bbox = gt_bbox

        # We probably don't need to do unions if there is sufficient padding
        # pred_bbox = parse_bbox_str(row['pred_bbox'])
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path).convert('RGB')


        # img is in xywh format
        cropped_img = np.array([])
        if read_imgs:
            cropped_img = crop_by_bbox_pil(img, bbox)
            cropped_img = resize(cropped_img, target_dim)
        # cropped_img.save(f'./clip_{idx}.png')
        # cropped_padding = crop_by_bbox_xyxy(img, do_padding_xyxy(img.size[::-1], bbox, 50))
        # cropped_padding = crop_by_bbox_xyxy(img, do_padding(img.size[::-1], bbox, 50))
        # cropped_padding.save(f'./clip_{idx}_padding.png')
        # raise Exception('saved')
        # bbox_xywh = bbox_xyxy_to_xywh(bbox)

        # The annotation id is replaced with index

        # Box already xywh
        bbox = [int(d) for d in bbox]
        img_data = ImageInfo(img_name, idx, cropped_img.flatten(), bbox)
        img_list.append(img_data)

        if n_imgs and len(img_list) >= n_imgs:
            break

    print(len(img_list), 'is len of images')
    
    return df[:len(img_list)], img_list

def preprocess_img_list(img_list):
    img_list = np.array(img_list)
    img_list_norm = (img_list - img_list.mean()) / img_list.std()
    return img_list_norm
    
def top_half(imgs):
    return imgs[:, :imgs.shape[1]//2]

def top_quarter(imgs):
    return imgs[:, :imgs.shape[1]//4]

def bot_half(imgs):
    return imgs[:, -imgs.shape[1]//2:]

def bot_quarter(imgs):
    return imgs[:, -imgs.shape[1]//4:]

def get_plugins():
    return {
        'top_half': top_half,
        'top_quarter': top_quarter,
        'bot_half': bot_half,
        'bot_quarter': bot_quarter,
    }

# ====================
def do_padding(img_dim, bbox_xywh, padding):
    '''
    Add padding to bounding box (xywh format).
    Arguments:
        img_dim: Tuple (w, h)
        bbox_xywh: Bounding box in xywh format
        padding: Either single padding in all directions or (w,h) padding
    '''
    # Padding is fixed, in all directions.
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x, y, x+w, y+h
    if isinstance(padding, int):
        x1 = max(0, x1 - padding)
        x2 = min(img_dim[1], x2 + padding)
        y1 = max(0, y1 - padding)
        y2 = min(img_dim[0], y2 + padding)
    elif len(padding) == 2:
        x1 = max(0, x1 - padding[0])
        x2 = min(img_dim[1], x2 + padding[0])
        y1 = max(0, y1 - padding[1])
        y2 = min(img_dim[0], y2 + padding[1])

    return x1, y1, x2, y2

def do_padding_xyxy(img_dim, bbox_xyxy, padding):
    x1, y1, x2, y2 = bbox_xyxy
    x, y, w, h = x1, y1, x2-x1, y2-y1
    return do_padding(img_dim, [x, y, w, h], padding)

def get_bbox(img, bbox_xywh, padding=None):
    '''
    get bbox for xywh format
    '''
    img_dim = img.shape[:2]
    if padding is not None:
        coords = do_padding(img_dim, bbox_xywh, padding)
    else:
        x, y, w, h = bbox_xywh
        coords = [x, y, x+w, y+h]

    # Coords are now in xyxy format
    coords = [int(c) for c in coords] 
    return img[coords[1]:coords[3], coords[0]:coords[2]]