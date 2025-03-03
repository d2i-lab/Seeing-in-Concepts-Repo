import os
import json
import argparse
from datetime import datetime
import subprocess
from multiprocessing import Pool
from time import time

import utils

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
import tqdm

WORKERS = 16
INTERACTION_METHOD = [
    'union',
    'either',
    'gt'
]

FILTER_METHOD = [
    'one-percent-classic',
]

def get_git_revision_hash() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode('ascii').strip()

def one_percent_filter_classic(segments):
    '''
    Method to filter out segments that are less than 1% of the image.
    * Remove all segments < 10 pixels 
    * Then return the indices of segments that are >= 1% of the image

    The multiple passes through the data is inefficient, but because of
    how we structured the data the first time, we have to follow the
    filtering exactly as it was done the first time.
    '''
    # segments = [seg for seg in segments if np.sum(seg) > 10]
    segment_sizes = [np.sum(seg) for seg in segments]
    segments = [seg for seg, size in zip(segments, segment_sizes) if size > 10]
    # merging_mask = np.zeros_like(segments[0])
    # # Fill empty mask with segment labels
    # for i, seg in enumerate(segments):
    #     merging_mask[np.nonzero(seg)] = i+1

    # # Unmerge segments, minus 1 to account for background
    # remaining_mask_ids = set(np.unique(merging_mask)) - set([0])
    # n_masks = remaining_mask_ids
    # new_segments = []
    # for i in n_masks:
    #     new_segments.append(merging_mask == i+1)

    with_id = []
    # mask_size_threshold = segments[0].size * 0.01
    mask_size_threshold = segments[0].size * 0.01
    for i, seg in enumerate(segments):
        if segment_sizes[i] >= mask_size_threshold:
            with_id.append((i, seg))
    return with_id


def interaction_union(gt_bbox_arr, pred_bbox_arr, mask, area_threshold=0.1,
                      scale=0.1, pixel_padding=0):
    '''
    Determine which segments are interacting with the bbox:
    * If both bboxes are present, use the union of the two.
    * If pred_bbox is empty, default to gt_bbox.
    * Else use pred_bbox.
    '''
    gt_bbox_exists = len(gt_bbox_arr) == 4
    pred_bbox_exists = len(pred_bbox_arr) == 4

    if gt_bbox_exists and pred_bbox_exists:
        bbox = utils.calculate_union(gt_bbox_arr, pred_bbox_arr)
    elif not pred_bbox_exists:
        bbox = gt_bbox_arr
    elif pred_bbox_exists:
        bbox = pred_bbox_arr

    if pixel_padding > 0 and scale <= 0:
        bbox = utils.do_padding(mask.shape, bbox, pixel_padding)

    bbox_subarray = utils.matrix_subarray(mask, bbox, scale=scale)
    bbox_area = bbox_subarray.shape[0] * bbox_subarray.shape[1]
    mask_area = np.count_nonzero(mask)
    overlap_area = np.count_nonzero(bbox_subarray)
    if area_threshold == 0 or area_threshold == None:
        # Count interaction as long as there is overlap
        return overlap_area > 0

    covers_bbox = overlap_area >= area_threshold * bbox_area
    covers_mask = overlap_area >= area_threshold * mask_area
    return covers_bbox or covers_mask

def interaction_gt(pred_bbox, mask, area_threshold=0.1, scale=0.1, pixel_padding=0):
    '''
    Determine which segments are interacting with gt bbox only
    '''
    if pixel_padding > 0 and scale <= 0:
        pred_bbox = utils.do_padding(mask.shape, pred_bbox, pixel_padding)

    bbox_subarray = utils.matrix_subarray(mask, pred_bbox, scale=scale)
    bbox_area = bbox_subarray.shape[0] * bbox_subarray.shape[1]
    mask_area = np.count_nonzero(mask)
    overlap_area = np.count_nonzero(bbox_subarray)
    if area_threshold == 0 or area_threshold == None:
        # Count interaction as long as there is overlap
        return overlap_area > 0

    covers_bbox = overlap_area >= area_threshold * bbox_area
    covers_mask = overlap_area >= area_threshold * mask_area
    return covers_bbox or covers_mask

def process_chunk(args):
    '''
    Process a chunk of the bbox CSV file.
    '''
    (chunk, sam_path, interaction_method, filter_method, padding) = args
    target_cols = ['gt_bbox_arr', 'pred_bbox_arr', 'img_path']
    sam_name = chunk['img_path'].iloc[0] + '.json'
    sam_path = os.path.join(sam_path, sam_name)

    masks = utils.extract_sam_mask(sam_path)
    if filter_method == 'one-percent-classic':
        masks = one_percent_filter_classic(masks)
    else:
        raise NotImplementedError('Filter method not implemented')

    new_rows = []
    for row in chunk[target_cols].itertuples():
        interacting_segments = []
        for (id, mask) in masks:
            if interaction_method == 'union':
                # TODO: Play around with area_threshold
                interacting = interaction_union(row.gt_bbox_arr, 
                                                row.pred_bbox_arr, mask, 
                                                area_threshold=0,
                                                scale=0, pixel_padding=padding)

            elif interaction_method == 'gt':
                interacting = interaction_gt(row.gt_bbox_arr, mask,
                                             area_threshold=0, 
                                             scale=0.1, pixel_padding=padding)
                                            #    area_threshold=0, scale=0.5)
            elif interaction_method == 'either':
                raise NotImplementedError('Interaction method not implemented')
            else:
                raise ValueError('Invalid interaction method')

            if interacting:
                interacting_segments.append(id)

        row_index = int(row.Index)
        new_rows.append([row_index, interacting_segments])

    return new_rows

def main(sam_path, bbox_path, interact_method, filter_method, padding, 
         out_path, limit, debug_mode, config_str, return_df=False):
    '''
    Function that retrieves bboxes from bbox_path (csv) and identifies which
    segments interact with the bbox. 

    Args:
        sam_path (str): Path to the directory containing the SAM jsons
        bbox_path (str): Path to the bbox CSV file
        interact_method (str): Method for determining which concepts are 
            counted as interacting
        filter_method (str): Segment filtering method. Make sure this 
            matches the method used in the first pass.
        out_path (str): Output file path
        limit (int): Limit the number of images to process
        debug_mode (bool): Debug mode. Prevents saving file
        config_str (str): JSON string of the config
        return_df (bool): Return the dataframe instead of saving to file

    Returns:
        df_joined (pd.DataFrame): Dataframe with the interacting segments
        also joined with the original bbox dataframe
    '''
    def _process_pool_queue(chunk_queue):
        with Pool(WORKERS) as pool:
            pool_result = pool.map(process_chunk, chunk_queue)
        return pool_result

    # Add compatibility to directly import dataframe as variable
    if isinstance(bbox_path, str):
        df = pd.read_csv(bbox_path)[:limit if limit else None]
        df['gt_bbox_arr'] = df['gt_bbox'].map(utils.parse_bbox_str)
        df['pred_bbox_arr'] = df['pred_bbox'].map(utils.parse_bbox_str)
    elif isinstance(bbox_path, pd.DataFrame):
        df = bbox_path
        df['gt_bbox_arr'] = df['gt_bbox']
        df['pred_bbox_arr'] = df['pred_bbox']

    if interact_method == 'gt':
        # Limit to df where bbox array is more than 0
        df = df[df['gt_bbox_arr'].map(len) > 0]
        print(f'Limiting to {len(df)} images with gt bbox')

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    img_index = {} # Maps image path to indexes
    for i, column in enumerate(df['img_path']):
        if column not in img_index:
            img_index[column] = []
        img_index[column].append(i)

    # Args that will be fed to process_chunk
    args = [sam_path, interact_method, filter_method, padding] 
    pool_queue = []
    for img in df['img_path'].unique().tolist():
        chunk = df.iloc[img_index[img]] # Chunk by image
        pool_queue.append((chunk, *args))

    unpacked_results = []
    for result in tqdm.tqdm(_process_pool_queue(pool_queue)):
        unpacked_results.extend(result)

    # Organize results
    out_cols = ['index', 'interacting_segments']
    result_df = pd.DataFrame(unpacked_results, columns=out_cols)

    if not return_df:
        result_df['interacting_segments'] = result_df['interacting_segments'].map(
            lambda x: np.array2string(np.array(x), precision=1, separator=','))

    result_df.set_index('index', inplace=True)
    result_df = result_df.rename_axis(None, axis=1)

    # We don't need these columns anymore
    df.drop(columns=['gt_bbox_arr', 'pred_bbox_arr'], inplace=True)
    # Join on common index
    df_joined = df.join(result_df)
    if return_df:
        return df_joined

    if not debug_mode:
        df_joined.to_csv(out_path)
    else:
        print(df_joined.head())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sam', type=str, required=True,
                        help='Path to the directory containing the SAM jsons')
    parser.add_argument('-o', '--out', type=str, required=True,
                        help='Output file path')
    parser.add_argument('-b', '--bbox', type=str, required=True, 
                        help='Path to the bbox CSV file')
    parser.add_argument('-i', '--interact', type=str, required=True,
                        choices=INTERACTION_METHOD,
                        help='Method for determining which concepts are counted as interacting')
    parser.add_argument('-f', '--filter', type=str, choices=FILTER_METHOD,
                        help='Segment filtering method. Make sure this matches the method used in the first pass.',
                        default=FILTER_METHOD[0])
    parser.add_argument('-p', '--padding', type=int, required=False, default=0,
                        help='Pixel padding for bbox. Default is 0')
    parser.add_argument('-l', '--limit', type=int, required=False,
                        help='Limit the number of images to process')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Debug mode. Prevents saving file')

    args = parser.parse_args()

    if not args.debug and os.path.exists(args.out):
        raise ValueError('Output file already exists')

    if not os.path.exists(args.sam):
        raise ValueError('SAM directory does not exist')

    if not os.path.exists(args.bbox):
        raise ValueError('Bbox CSV file does not exist')

    config = vars(args)
    config['time'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    config['git-hash'] = get_git_revision_hash()
    config_str = json.dumps(config)

    main(args.sam, args.bbox, args.interact, args.filter, args.padding, args.out, 
         args.limit, args.debug, config_str)