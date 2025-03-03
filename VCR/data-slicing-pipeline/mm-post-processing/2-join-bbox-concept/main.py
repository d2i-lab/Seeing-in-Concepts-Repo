import os
import json
import argparse
from datetime import datetime
import subprocess
from multiprocessing import Pool, Manager
from time import time

import utils

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
import tqdm

CHUNK_SIZE = 2e3
WORKERS = 32

# 
# concept -> img -> segment idx

def get_git_revision_hash() -> str:
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']
    ).decode('ascii').strip()

def get_labels(segment_path)->list:
    label_path = os.path.join(segment_path, 'labels.txt')
    if not os.path.exists(label_path):
        raise Exception('Cannot find dataset labels')
    
    with open(label_path, 'r') as f:
        l = f.read()

    return l.split(',')

def process_chunk(args):
    (chunk, len_all_labels, labels_to_id, include_FN, segment_path, image_folder, 
     save_bbox, count_concepts, img_to_concept, img_to_seg) = args

    concept_data = []
    img_path = []
    gt_label = []
    pred_label = []
    gt_bbox = []
    pred_bbox = []
    ignore_flag = []
    pred_score = []
    fp_type1 = []
    fp_type2 = []
    fn = []
    iou = []

    gt_crowd = []
    pred_crowd = []
    gt_confusion = []
    pred_confusion = []

    chunk['gt_bbox_arr'] = chunk['gt_bbox'].map(utils.parse_bbox_str)
    chunk['pred_bbox_arr'] = chunk['pred_bbox'].map(utils.parse_bbox_str)
    # has_ignore_flag = 'ignore_flag' in list(chunk.columns)
    print('Starting')
    for row in chunk.itertuples():
        bbox = utils.get_bbox(row.gt_bbox_arr, 
                                row.pred_bbox_arr, include_FN)
        if bbox is None:
            continue

        if count_concepts:
            if img_to_concept is None:
                concepts = utils.find_concepts_path_count(segment_path, bbox,
                                                        image_folder, row.img_path)
            else:
                # Remap:
                try:
                    concepts = utils.find_concepts_SAM_remap(segment_path, bbox, 
                                                            row.img_path, 
                                                            img_to_concept,
                                                            img_to_seg, 
                                                            count=count_concepts)
                except Exception as e:
                    print(e)
                    continue

            # label_ids = [labels_to_id[label] for label in concepts.keys()]
            # concept_row = np.zeros(len(all_labels)).astype(np.uint8)
            concept_row = np.zeros(len_all_labels).astype(np.uint8)
            # concept_row = np.zeros(532).astype(np.uint8)
            for label, count in concepts.items():
                concept_row[labels_to_id[label]] = count
                # concept_row[label] = count
            
            # print(np.count_nonzero(concept_row), np.sum(concept_row))
            concept_row = csr_matrix(concept_row)
        else:
            if img_to_concept is None:
                concepts = utils.find_concepts_path(segment_path, bbox, 
                                                    image_folder, row.img_path)
            else:
                # Remap:
                # try:
                concepts = utils.find_concepts_SAM_remap(segment_path, bbox, 
                                                        row.img_path,
                                                        img_to_concept,
                                                        img_to_seg, 
                                                        count=False)
                if len(concepts) == 0:
                    continue

                # except Exception as e:
                    # print(e)
                    # continue

            # concept_row = np.zeros(len(all_labels)).astype(bool)
            concept_row = np.zeros(len_all_labels).astype(bool)
            label_ids = [labels_to_id[label] for label in concepts]
            concept_row[label_ids] = 1 # TODO: Allow for continuous val
            concept_row = csr_matrix(concept_row)

        concept_data.append(concept_row)
        img_path.append(row.img_path)
        gt_label.append(row.gt_label)
        pred_label.append(row.pred_label)

        # print(type(row), row, chunk.columns)
        # if has_ignore_flag:
        ignore_flag.append(row.ignore_flag)

        if save_bbox:
            gt_bbox.append(row.gt_bbox) # Note, we are appending string bbox
            pred_bbox.append(row.pred_bbox) # Appending string bbox as well

        pred_score.append(row.pred_score)
        fp_type1.append(row.fp_type1)
        fp_type2.append(row.fp_type2)
        fn.append(row.fn)
        iou.append(row.iou)

        gt_crowd.append(row.gt_crowding)
        pred_crowd.append(row.pred_crowding)
        gt_confusion.append(row.gt_confusion)
        pred_confusion.append(row.pred_confusion)

    return [concept_data, img_path, gt_label, pred_label, gt_bbox, pred_bbox, ignore_flag, pred_score, fp_type1, fp_type2, fn, iou] + [gt_crowd, pred_crowd, gt_confusion, pred_confusion]

def main(out_file, config_str, bbox_path, segment_path, image_folder, segment_dtype, 
        include_FN, save_bbox, count_concepts, remap_file):


    remap_json = None
    img_to_concept = None
    img_to_seg = None
    if not remap_file:
        all_labels = get_labels(segment_path)
    else:
        with open(remap_file, 'r') as f:
            remap_json = json.load(f)
        all_labels = list(remap_json['cluster_map'].keys())
        # Rename label id's to string names
        # all_labels = [remap_json['cluster_names'][label] for label in all_labels]
        # print(remap_json.keys())
        img_to_concept, img_to_seg = utils.get_concept_per_image(remap_json)
        

    labels_to_id = {label:i for i, label in enumerate(all_labels)}
    print(labels_to_id.keys())
    # If the segment is a single JSON file, we need to handle it differently
    if segment_dtype == 'json':
        raise NotImplementedError

    with Manager() as manager:
        if img_to_concept:
            img_to_concept = manager.dict(img_to_concept)
            img_to_seg = manager.dict(img_to_seg)
        with pd.read_csv(bbox_path, chunksize=CHUNK_SIZE, iterator=True) as reader:
            config = [len(all_labels), labels_to_id, include_FN, segment_path, 
                    image_folder, save_bbox, count_concepts, img_to_concept,
                    img_to_seg]
            results = []
            chunk_queue = []
            for chunk in reader:
                chunk_queue.append((chunk, *config))
                if len(chunk_queue) >= WORKERS*2:
                    # map -> [[iou, asd], [iou, asd]]
                    #print('Processing chunk')
                    #print(len(chunk_queue))
                    print('Staring new pool')
                    t = time()
                    with Pool(processes=WORKERS) as pool:
                        results.extend(pool.map(process_chunk, chunk_queue))
                    print('Pool took', time() - t)
                    chunk_queue = []

            if len(chunk_queue) > 0:
                print('Starting extended pool', len(chunk_queue), len(chunk_queue))
                with Pool(processes=WORKERS) as pool:
                    results.extend(pool.map(process_chunk, chunk_queue))

            concept_data = []
            img_path = []
            gt_label = []
            pred_label = []
            gt_bbox = []
            pred_bbox = []
            ignore_flag = []
            pred_score = []
            fp_type1 = []
            fp_type2 = []
            fn = []
            iou = []

            gt_crowd = []
            pred_crowd = []
            gt_confusion = []
            pred_confusion = []
            for chunk in results:
                (_concenpt_data, _img_path, _gt_label, _pred_label, _gt_bbox, 
                _pred_bbox, _ignore_flag, _pred_score, _fp_type1, _fp_type2, _fn, _iou,
                _gt_crowd, _pred_crowd, _gt_confusion, _pred_confusion) = chunk
                concept_data.extend(_concenpt_data)
                img_path.extend(_img_path)
                gt_label.extend(_gt_label)
                pred_label.extend(_pred_label)
                gt_bbox.extend(_gt_bbox)
                pred_bbox.extend(_pred_bbox)
                ignore_flag.extend(_ignore_flag)
                pred_score.extend(_pred_score)
                fp_type1.extend(_fp_type1)
                fp_type2.extend(_fp_type2)
                fn.extend(_fn)
                iou.extend(_iou)

                gt_crowd.extend(_gt_crowd)
                pred_crowd.extend(_pred_crowd)
                gt_confusion.extend(_gt_confusion)
                pred_confusion.extend(_pred_confusion)

    iou = np.array(iou, dtype=np.float16)
    sparse_matrix = vstack(concept_data)
    npz_str = utils.sparse_to_npz(sparse_matrix)
    data_pack = utils.DataPackage(
        img_path=img_path,
        gt_label=gt_label,
        pred_label=pred_label,
        gt_bbox=gt_bbox,
        pred_bbox=pred_bbox,
        ignore_flag=ignore_flag,
        pred_score=pred_score,
        fp_type1=fp_type1,
        fp_type2=fp_type2,
        fn=fn,
        iou=iou,
        gt_crowd=gt_crowd,
        pred_crowd=pred_crowd,
        gt_confusion=gt_confusion,
        pred_confusion=pred_confusion
    )
    # print(data_pack.img_path[:5], data_pack.gt_label[:5], data_pack.pred_label[:5], data_pack.pred_score[:5])
    #print(data_pack.fp_type1[:5], data_pack.fp_type2[:5], data_pack.fn[:5], data_pack.iou[:5])
    print(data_pack.gt_bbox[:5], data_pack.pred_bbox[:5])
    if remap_file:
        all_labels = [remap_json['cluster_names'][label] for label in all_labels]

    utils.save_to_hdf5(out_file, config_str, npz_str, all_labels, data_pack)

if __name__ == '__main__':
    dtype_choices = [
        'json_folder',
        'json'
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('-seg', '--segment_dataset', type=str, required=True)
    parser.add_argument('-stype', '--segment_dataset_type', type=str, 
                        choices=dtype_choices, required=True)
    parser.add_argument('-bbox', '--bbox_dataset', type=str, required=True)
    parser.add_argument('-o', '--out', type=str, required=True)
    parser.add_argument('-fn', '--false_negative', action='store_true', 
                        default=False)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--save_bbox', action='store_true')
    parser.add_argument('--count_concepts', action='store_true')
    parser.add_argument('--remap', type=str, 
                        help='Segment->Concept remapping file')
    args = parser.parse_args() 

    config = vars(args)
    config['time'] = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    config['git-hash'] = get_git_revision_hash()

    if os.path.exists(args.out + '.hdf5'):
        raise Exception('Out file already exists')

    config_str = json.dumps(config)
    main(args.out, config_str, args.bbox_dataset, args.segment_dataset, 
        args.image_folder, args.segment_dataset_type, args.false_negative, 
        args.save_bbox, args.count_concepts, args.remap)
