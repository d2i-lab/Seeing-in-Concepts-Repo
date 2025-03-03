import os
import json
import base64
import subprocess
from io import BytesIO
from dataclasses import dataclass

import h5py
import numpy as np
import pycocotools
import pycocotools.mask
from PIL import Image
from scipy.sparse import csr_matrix, vstack, save_npz 

@dataclass
class DataPackage:
    img_path: list
    gt_label: list
    pred_label: list
    gt_bbox: list # List of arrays as strings (e.g. ['123,123,123,123',...])
    pred_bbox: list # List of arrays as strings (e.g. ['123,123,123,123',...])
    ignore_flag: list
    pred_score: list
    fp_type1: list
    fp_type2: list
    fn: list
    iou: list

    gt_crowd: list
    pred_crowd: list
    gt_confusion: list
    pred_confusion: list

def get_git_revision_hash() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode('ascii').strip()

def parse_bbox_str(bbox_str, dtype=float):
   return np.fromstring(bbox_str[1:-1], dtype=dtype, sep=',') 

def calculate_union(box1, box2):
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    
    union_box = [x_min, y_min, x_max, y_max]
    return union_box

def matrix_subarray(full_matrix, bbox, scale=.1):
    h = bbox[3] - bbox[1]
    w = bbox[2] - bbox[0]
    padding_row = int(w * scale/2)
    padding_col = int(h * scale/2)
    row1 = max(int(bbox[1])-padding_row, 0)
    row2 = min(int(bbox[3])+padding_row, full_matrix.shape[0])
    col1 = max(int(bbox[0])-padding_col, 0)
    col2 = min(int(bbox[2])+padding_col, full_matrix.shape[1])
    img_bbox = full_matrix[row1:row2,col1:col2]
    return img_bbox

def do_padding(img_dim, bbox_xyxy, padding):
    '''
    Add padding to bounding box (xywh format).
    Arguments:
        img_dim: Tuple
        bbox_xywh: Bounding box in xywh format
        padding: Either single padding in all directions or (w,h) padding
    '''
    # Padding is fixed, in all directions.
    x1, y1, x2, y2 = bbox_xyxy
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

#TODO: Use in get_bbox maybe
def scale_bbox_with_aspect_ratio(bbox, scale_factor, img_shape):
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Calculate the original width and height of the bounding box
    original_width = x2 - x1
    original_height = y2 - y1

    # Calculate the center point of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the aspect ratio of the original bounding box
    aspect_ratio = original_width / original_height

    # Calculate the new width and height based on the desired scaling factor and aspect ratio
    new_width = original_width * scale_factor
    new_height = new_width / aspect_ratio

    # Calculate the new coordinates of the scaled bounding box
    new_x1 = max(int(center_x - new_width / 2), 0)
    new_y1 = max(int(center_y - new_height / 2), 0)
    new_x2 = min(int(center_x + new_width / 2), img_shape[1])
    new_y2 = min(int(center_y + new_height / 2), img_shape[0])

    return new_x1, new_y1, new_x2, new_y2


def get_bbox(gt_bbox, pred_bbox, include_FN, do_union=True):
    gt_bbox_exists = len(gt_bbox) == 4
    pred_bbox_exists = len(pred_bbox) == 4
    bbox = None
    if gt_bbox_exists and pred_bbox_exists:
        # Both GT and DT bboxes are present
        # TODO: Figure out if unioning bboxes is the best approach
        bbox = gt_bbox
        if do_union:
            bbox = calculate_union(gt_bbox, pred_bbox)

    elif gt_bbox_exists and not pred_bbox_exists:
        # False Negative: GT exists, but not prediction
        if not include_FN:
            return None
        bbox = gt_bbox
    elif not gt_bbox_exists and pred_bbox_exists:
        # False Positive: Prediction exits, GT does not
        bbox = pred_bbox
    
    return bbox

def should_add_concept(mask, bbox, area_threshold=0.1, scale=0.1):
    bbox_subarray = matrix_subarray(mask, bbox, scale)
    bbox_area = bbox_subarray.shape[0] * bbox_subarray.shape[1]

    # Consider concept present if: 
    #   1) Mask covers more than area_threshold of bbox 
    #   2) More than area_threshold is present in bbox
    mask_area = np.count_nonzero(mask)
    mask_bbox_area = np.count_nonzero(bbox_subarray)
    covers_bbox = mask_bbox_area >= area_threshold * bbox_area
    covers_mask = mask_bbox_area >= area_threshold * mask_area

    # TODO: Can play around with this condition.
    return covers_bbox or covers_mask

def decode_mask(some_mask, img_folder, img_path):
    mask = None
    if isinstance(some_mask, str):
        # TODO: Figure out if there's a faster way.
        # This method only exists because I accidentally removed the size
        # parameter when encoding the mask data.
        with Image.open(os.path.join(img_folder, img_path)) as im:
            rows, cols = im.size[::-1] # (w,h) -> (h,w)
        mask = {'counts': some_mask, 'size': [rows, cols]}
    elif isinstance(some_mask, dict):
        mask = some_mask # Place-holder
    else:
        print('Unexpected mask type: {}'.format(type(some_mask)))
        exit(1)

    mask = pycocotools.mask.decode(mask) 
    return mask

def extract_sam_mask(json_file):
    '''
    Extract SAM masks.
    '''
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    segments = [seg['segmentation'] for seg in json_dict]
    decoded_segments = []
    for seg in segments:
        decoded_segments.append(pycocotools.mask.decode(seg))

    return decoded_segments

def get_concept_per_image(remap):
    '''
    Given a remap, return a dictionary mapping image names to a list of
    (seg_id, concept_id) tuples.
    '''
    img_to_concept = {} # Img Name -> [Concept ID]
    img_to_seg = {} # Img Name -> [Seg ID]
    for cluster_id in remap['cluster_map']:
        for img_name in remap['cluster_map'][cluster_id]:
            if img_name not in img_to_concept:
                img_to_concept[img_name] = []
                img_to_seg[img_name] = []
            for seg_id in remap['cluster_map'][cluster_id][img_name]:
                img_to_concept[img_name].append(cluster_id)
                img_to_seg[img_name].append(seg_id)
    return img_to_concept, img_to_seg

def find_concepts_SAM_remap(segment_path, bbox, img_path, 
                            img_to_concept, img_to_seg, count=False, 
                            area_threshold=.1)->list:
    '''
    Finds concepts in the SAM remapped masks. 
    If count=True, returns a dictionary mapping concept_id to count.
    Otherwise, returns a list of concept_ids.
    '''
    sam_file = os.path.join(segment_path, '{}.json'.format(img_path))
    if not os.path.exists(sam_file):
        # raise Exception('Cannot find masks file: {}'.format(sam_file))
        print('Cannot find masks file: {}'.format(sam_file))
        return {}

    segments = extract_sam_mask(sam_file)
    # Apparently this line is super expensive. It's worth half the runtime.
    segments = [seg for seg in segments if np.sum(seg) > 10]
    img_name = img_path.split('.')[0]
    if img_name not in img_to_concept:
        # raise Exception('Cannot find image in remap: {}'.format(img_name))
        print('Cannot find image in remap: {}'.format(img_name))
        return {}

    concepts = {}
    # seg_ids, concept_ids = remap_img_concept[img_name]
    seg_ids = img_to_seg[img_name]
    concept_ids = img_to_concept[img_name]
    for seg_id, concept_id in zip(seg_ids, concept_ids):
        # Subtract 1 because seg_id's generated from export are 1-indexed
        mask = segments[seg_id - 1]
        if should_add_concept(mask, bbox, area_threshold):
            concepts[concept_id] = concepts.get(concept_id, 0) + 1

    if count:
        return concepts
    
    return list(concepts.keys())

def find_concepts_path(segment_path, bbox, img_folder, img_path, area_threshold=.1)->list:
    json_file = '{}-masks.json'.format(img_path)
    masks_file = os.path.join(segment_path, json_file)
    if not os.path.exists(masks_file):
        raise Exception('Cannot find masks file: {}'.format(masks_file))
        # return []

    mask_types = None
    with open(masks_file, 'r') as f:
        try: # Spooky json load. Wrap in try catch
            mask_types = json.load(f)
        except:
            print('Failed to load:', masks_file) # TODO: Use logger ¯\_(ツ)_/¯
    
    if not mask_types:
        raise Exception('Failed to load masks file: {}'.format(masks_file))
        # return []

    concepts = set()
    for mask_type_name in ['panoptic', 'part']:
        (encoded_masks, labels) = mask_types[mask_type_name]
        for (mask, label) in zip(encoded_masks, labels):
            mask = decode_mask(mask, img_folder, img_path)
            if should_add_concept(mask, bbox, area_threshold):
                concepts.add(label)

    return list(concepts)

def find_concepts_path_count(segment_path, bbox, img_folder, img_path, area_threshold=.1)->list:
    json_file = '{}-masks.json'.format(img_path)
    masks_file = os.path.join(segment_path, json_file)
    if not os.path.exists(masks_file):
        json_file = '{}.json'.format(img_path)
        masks_file = os.path.join(segment_path, json_file)
        if not os.path.exists(masks_file):
            raise Exception('Cannot find masks file: {}'.format(masks_file))
            # return {}
        # return {}

    mask_types = None
    with open(masks_file, 'r') as f:
        try: # Spooky json load. Wrap in try catch
            mask_types = json.load(f)
        except:
            print('Failed to load:', masks_file) # TODO: Use logger ¯\_(ツ)_/¯
    
    if not mask_types:
        raise Exception('Failed to load masks file: {}'.format(masks_file))
        # return {}

    concepts = {}
    for mask_type_name in ['panoptic', 'part']:
        (encoded_masks, labels) = mask_types[mask_type_name]
        for (mask, label) in zip(encoded_masks, labels):
            mask = decode_mask(mask, img_folder, img_path)
            if should_add_concept(mask, bbox, area_threshold):
                concepts[label] = concepts.get(label, 0) + 1

    return concepts

def sparse_to_npz(mat):
    buffer = BytesIO()
    save_npz(buffer, mat)
    buffer.seek(0)
    npz_str = buffer.getvalue()
    buffer.close()
    return npz_str

# def save_to_hdf5(out_file, config_str, npz_str, ious, labels):
def save_to_hdf5(out_file, config_str, npz_str:str, labels, 
                 data_pack:DataPackage, concept_mapping:dict=None): 
    # Encode again because .decode('utf-8') does not like null-bytes
    # Can possibly use .decode('latin-1'), but seems sketchy
    encoded_npz = base64.b64encode(npz_str)
    g = 'gzip'
    #print('THis is ignore flag', data_pack.ignore_flag)
    with h5py.File('{}.hdf5'.format(out_file), 'w') as f:
        f.create_dataset('sparse_matrix', data=[encoded_npz]) # Must be in array
        f.create_dataset('labels', data=labels, compression=g)
        f.create_dataset('img_paths', data=data_pack.img_path, compression=g)
        f.create_dataset('gt_labels', data=data_pack.gt_label, compression=g)
        f.create_dataset('pred_labels', data=data_pack.pred_label, compression=g)
        f.create_dataset('gt_bboxes', data=data_pack.gt_bbox, compression=g)
        f.create_dataset('pred_bboxes', data=data_pack.pred_bbox, compression=g)
        f.create_dataset('ignore_flags', data=data_pack.ignore_flag, compression=g)
        f.create_dataset('pred_scores', data=data_pack.pred_score, compression=g)
        f.create_dataset('fp_type1s', data=data_pack.fp_type1, compression=g)
        f.create_dataset('fp_type2s', data=data_pack.fp_type2, compression=g)
        f.create_dataset('fns', data=data_pack.fn, compression=g)
        f.create_dataset('ious', data=data_pack.iou, compression=g)

        f.create_dataset('gt_crowds', data=data_pack.gt_crowd, compression=g)
        f.create_dataset('pred_crowds', data=data_pack.pred_crowd, compression=g)
        f.create_dataset('gt_confusions', data=data_pack.gt_confusion, compression=g)
        f.create_dataset('pred_confusions', data=data_pack.pred_confusion, compression=g)
        f.attrs['config'] = config_str

        if concept_mapping:
            concept_mapping_json = json.dumps(concept_mapping)
            f.attrs['concept_mapping'] = concept_mapping_json
