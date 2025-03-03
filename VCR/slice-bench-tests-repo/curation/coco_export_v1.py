import os
import sys
import json
import argparse
import warnings
import contextlib
import pathlib
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score

from divexplorer3.divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer3.divexplorer.FP_Divergence import FP_Divergence


# Change directory to mm-post-processing/2-join-bbox-concept
sys.path.append('../../data-slicing-pipeline/mm-post-processing/2-join-bbox-concept')
print(sys.path)
import bbox_segments

sys.path.append('..')
import cliqueminer.strat.clique2 as clique2
sys.path.append('../../cluster-explorer/backend')
import api.logic.cluster_handler as cluster_handler
import api.logic.export_handler as export_handler
import api.utils.settings as settings
import api.utils.query as query


COLUMNS = [
    'gt_bbox', 'pred_bbox', 'img_path', 
]

# EVAL_K = [10, 25]
# EVAL_K = [10, 25]
EVAL_K = [10]
N_SLICES = 10

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return np.array([x, y, x+w, y+h])

def count_nonzero_interactions(df):
    count = 0
    for interaction in df['interacting_segments']:
        if len(interaction) > 0:
            count += 1
    return count / len(df)

def create_concept_mapping(settings_obj, k_concepts):
    cluster_handler_obj = cluster_handler.ClusterHandler(settings_obj)
    cluster_data = cluster_handler_obj.cluster_embeddings(
        # ncentroids=200, 
        ncentroids=k_concepts, 
        dimensionality=-1, 
        embeddingsType='CLIP'
    )
    cluster_map = cluster_data['cluster_map']
    cluster_names = {}
    for cid in cluster_map:
        cluster_names[str(cid)] = str(cid)

    return cluster_map, cluster_names

def divexplore_fit(dataframe, min_support, max_len, th_redundancy=None, include_absence=True, FPM_type='fpgrowth'):
    """
    Run divexplorer given dataframe and other params.
    
    Returns:
    - result_divexplore: Dataframe with itemsets and other info.
    - fp_divergence_acc: Data containing post-mining results.
    """
    negative = 0
    positive = 1
    true_class, predicted_class = 'true', 'predicted'
    df = dataframe.copy()

    fp_diver = FP_DivergenceExplorer(df, true_class, predicted_class, class_map={"P":positive, "N":negative}, include_absence=include_absence)
    dummy_df = fp_diver.X.copy() # Adds one-hot-encoded columns

    FP_fm = fp_diver.getFrequentPatternDivergence(min_support=min_support, metrics=["d_accuracy"],max_len=max_len, FPM_type=FPM_type)
     
    # Fitler on d_accuracy and run de-dup functions (if th_redundancy != None)
    result_divexplore=FP_fm
    fp_divergence_acc=FP_Divergence(FP_fm, "d_accuracy")
    result_divexplore=fp_divergence_acc.getDivergence(th_redundancy=th_redundancy)
    result_divexplore = result_divexplore.sort_values(by=['d_accuracy'], ascending=True)
    return result_divexplore, fp_divergence_acc, dummy_df

def prepare_greedy_dedup(res_df, ori_df, top_k=50, max_overlap=0.5):

    # TODO: Turn this into a bitmap
    covered_idx = set()

    def _get_itemset(idx):
        itemset_str = query.itemset_to_str(res_df.loc[idx]['itemsets'])
        df_result = set(query.get_slice(ori_df, itemset_str).index)
        return df_result

    print('running dedup')
    display_idx = []
    for idx, row in res_df.iterrows():
        i1 = _get_itemset(idx)
        covered_intersection = set(covered_idx).intersection(i1)
        if len(covered_intersection) / len(i1) <= max_overlap:
            display_idx.append(idx)
            covered_idx.update(i1)
        if len(display_idx) >= top_k:
            break
    
    new_res_df = res_df.loc[display_idx]
    return new_res_df.sort_values(by='d_accuracy')

def run_miner(df, min_support, iou, absence=False, dedup=0.5, top_k=10):
    df['Outlier'] = (iou < 0.5).astype(bool)
    df['predicted'] = ~df['Outlier']
    df['true'] = True
    df = df.drop(columns=['Outlier'])

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                results, _, dummy_df = divexplore_fit(
                    df, min_support, max_len=3, th_redundancy=None,
                    include_absence=absence,
                )

    results['itemsets'] = results['itemsets'].apply(lambda x: tuple(x))
    results = results.rename(columns={'support_count': 's_count'})
    results = results[['itemsets', 's_count', 'support', 'accuracy', 'd_accuracy']]

    if results.shape[0] < 10:
        warnings.warn('Not enough results: {}'.format(results.shape[0]))

    if dedup < 0:
        print('No De-dup')
        return results 

    results = prepare_greedy_dedup(results, dummy_df, top_k, dedup)
    return results

def run_single_test(n_slices, df, logits, gt_slice, absence=False, support=max(EVAL_K), dedup=0.5):
    iou = np.array(logits) # We're using the logits as the iou for now

    precisions = {}

    real_min_support = max([10]) / len(gt_slice) - 1e-6
    if support >= 1:
        support = int(support)
        min_support = support / len(gt_slice) - 1e-6
    else:
        min_support = support

    min_support = max(min_support, real_min_support)
    


    results = run_miner(df, min_support, iou, absence=absence, dedup=dedup)
    print('--'*15)

    covered_percents = {}
    purity = {}
    print(results)

    # for k in EVAL_K:
    for k in [10]:
        pk = []
        for i in range(min(n_slices, len(results))):
            itemsets = results.iloc[i]['itemsets']
            itemset_list = list(itemsets)


            # FutureWarning: The behavior of .astype from SparseDtype to a 
            # non-sparse dtype is deprecated. In a future version, this will 
            # return a non-sparse array with the requested dtype. To retain the 
            # old behavior, use `obj.astype(SparseDtype(dtype))`
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df_sliced = query.query_by_list(df, itemset_list)

            percent_gt_in_slice = sum(gt_slice[df_sliced.index]) / sum(gt_slice)
            covered_percents[i] = percent_gt_in_slice
            purity[i] = sum(gt_slice[df_sliced.index]) / len(df_sliced)

            # Get the indices of lowest k ious
            prediction = np.zeros(len(gt_slice), dtype=bool)
            if len(df_sliced) > k:
                df_sliced_ious = iou[df_sliced.index]
                indices = df_sliced.index[np.argpartition(df_sliced_ious, k)[:k]]
            elif len(df_sliced) == k:
                indices = df_sliced.index
            else:
                raise Exception('This should not happen')

            prediction[indices] = True

            pk.append(precision_score(gt_slice, prediction))
        precisions[k] = pk

    print('covered_percents', covered_percents)
    print('purity', purity)
    
    return precisions

def main(json_path, sam_folder, img_folder, embed_folder, fastpath_pkl, 
         interact_method, filter_method, padding, k_concepts, allow_absence,
         allow_box_area, allow_box_aspect, no_concepts,
         skip_low_err,
         out_path, config_str,
         support=max(EVAL_K), dedup=0.5
         ):
    with open(json_path) as f:
        data = json.load(f)
    
    img_name = data['img_name']
    cluster_label = data['cluster_label']
    bboxes = data['bboxes']

    if 'label_id' in data:
        class_label = data['label_id']
    else:
        # Need to convert
        class_label = data['cat_id'] - 1

    bboxes_converted = [xywh_to_xyxy(b) for b in bboxes]
    pred_bboxes = [np.array([]) for _ in range(len(bboxes_converted))]

    df = pd.DataFrame({
        'gt_bbox': bboxes_converted,
        'pred_bbox': pred_bboxes,
        'img_path': img_name,
        'slice_bucket': cluster_label 
    })
    # TODO: If injection method is label missclassification, then we need to
    # change the pred label per slice bucket
    df['pred_label'] = class_label
    df['gt_label'] = class_label


    
    if not no_concepts:
        interact_df_original = bbox_segments.main(sam_folder, df, interact_method, 
                                        filter_method, padding, out_path, None,
                                        debug_mode=False, config_str=config_str, 
                                        return_df=True)

        # Create concept mapping
        out_folder = os.path.dirname(out_path) or '.'
        out_file_name = os.path.basename(out_path)
        settings_obj = settings.Settings(
            coco_img_dir = img_folder,
            sam_jsons_dir = sam_folder,
            seg_embeddings_dir = embed_folder,
            remap_output_dir = out_folder,
            fast_path = fastpath_pkl
        )
        cluster_map, cluster_names = create_concept_mapping(settings_obj, k_concepts)
        # Create concept_df using interact_df and cluster_map
        export_handler_obj = export_handler.ExportHandler(settings_obj)
        concept_df = export_handler_obj.export({
            'exportDict': {
                'metadata': {
                    'fname': out_file_name,
                    'segment_bbox_path': sam_folder,
                },
                'cluster_map': cluster_map,
                'cluster_names': cluster_names,
            }
        }, interact_df=interact_df_original, return_df=True)

        bins = 5
        if allow_box_area:
            print('allowing box area')
            concept_df['box_area'] = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes_converted]
            concept_df['box_area'] = pd.qcut(concept_df['box_area'], bins, labels=False)
        if allow_box_aspect:
            print('allowing box aspect')
            concept_df['box_aspect'] = [(b[2]-b[0])/(b[3]-b[1]) for b in bboxes_converted]
            concept_df['box_aspect'] = pd.qcut(concept_df['box_aspect'], bins, labels=False)
    else:
        bins = 5
        print('NO CONCPETS')
        box_area = [(b[2]-b[0])*(b[3]-b[1]) for b in bboxes_converted]
        box_aspect = [(b[2]-b[0])/(b[3]-b[1]) for b in bboxes_converted]
        concept_df = pd.DataFrame({
            'box_area': box_area,
            'box_aspect': box_aspect,
        })
        concept_df['box_area'] = pd.qcut(concept_df['box_area'], bins, labels=False)
        concept_df['box_aspect'] = pd.qcut(concept_df['box_aspect'], bins, labels=False)

    # TODO: ??
    # if allow_crowding:
    #     pass

    precisions = {}
    for error_setting in tqdm(data['pred_logits'], desc='Error settings'):
        precisions[error_setting] = {}
        if error_setting == 'low' and skip_low_err:
            print('Skipping low error test')
            continue
        for problem_label in tqdm(data['pred_logits'][error_setting], desc='Problem labels'):

            # Create array representing whether each sample is in the slice
            if problem_label.isnumeric():
                gt_slice = np.array(data['cluster_label']) == int(problem_label)
            else:
                gt_slice = np.array(data['cluster_label']) == problem_label

            n_in_slice = np.sum(gt_slice)
            if any([n_in_slice < k for k in [25]]):
                warnings.warn(f'Not enough samples in slice {problem_label}')
                continue

            print('running test', error_setting, problem_label)
            logits = data['pred_logits'][error_setting][problem_label]
            pk = run_single_test(N_SLICES, concept_df, logits, gt_slice, absence=allow_absence, support=support, dedup=dedup)
            precisions[error_setting][problem_label] = pk

    # Save metadata to file
    label_count = Counter(data['cluster_label'])
    precisions['args_dict'] = config_str
    precisions['label_count'] = label_count
    with open(out_path+'.json', 'w') as f:
        json.dump(precisions, f)
    
    return precisions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', type=str, 
                        help='JSON file containing test data')
    parser.add_argument('--json_folder', '-jf', type=str, 
                        help='Path to the directory containing the JSON files')
    parser.add_argument('--sam', '-s', type=str, required=True,
                        help='Path to the directory containing the SAM jsons')
    parser.add_argument('--img', '-im', type=str, required=True,
                        help='Path to the directory containing the images')
    parser.add_argument('--embed', '-e', type=str, required=True,
                        help='Path to the directory containing the embeddings')
    parser.add_argument('--fastpath', '-fp', type=str,
                        help='Path to the fastpath embeddings pickle')
    parser.add_argument('--out', '-o', type=str,
                        help='Output file path')
    parser.add_argument('--interact', '-i', type=str, required=True,
                        choices=bbox_segments.INTERACTION_METHOD, 
                        help='Interaction method to use', default='gt')
    parser.add_argument('--filter', '-f', type=str, required=True, 
                        choices=bbox_segments.FILTER_METHOD,
                        help='Limit the number of images to process')
    parser.add_argument('--padding', '-p', type=int, default=0,
                        help='Padding for bounding box')
    parser.add_argument('--k_concepts', '-k', type=int, default=200,
                        help='Number of concepts to form with k-means.')
    parser.add_argument('--support', nargs='+', type=float, 
                        help='Support value(s) to run at. Support < 1 signifies support % while > 1 will do support count. e.g. support 0.25 represents 25% while support 25 represents count of 25.')
    parser.add_argument('--dup_thresh', nargs='+', type=float,
                        help='Dedup threshold(s) to run at from range (0,1). Lower value -> more aggressive deduplication')
    parser.add_argument('--allow_absence', '-aa', action='store_true',
                        help='Allow absence in the miner')
    parser.add_argument('--allow_box_area', '-aba', action='store_true',)
    parser.add_argument('--allow_box_aspect', '-abasp', action='store_true',)
    parser.add_argument('--no_concepts', '-nc', action='store_true',
                        help='Do not form concepts with k-means')
    parser.add_argument('--skip_low_err', action='store_true', default=False,
                        help='Whether or not to run low error tests. In our evaluation, we ignore these.')
    parser.add_argument('--force', '-fo', action='store_true',
                        help='Force overwrite of output file')
    args = parser.parse_args()
    config_str = json.dumps(vars(args))

    if args.json:
        if not args.out:
            raise ValueError('Must provide output file path')

        if os.path.exists(args.out+'.json'):
            raise ValueError(f'Output file {args.out}.json already exists')

        main(args.json, args.sam, args.img, args.embed, args.fastpath, 
            args.interact, args.filter, args.padding, args.k_concepts,
            args.allow_absence, args.allow_box_area, args.allow_box_aspect,
            args.no_concepts,
            args.skip_low_err,
            args.out, config_str)

    elif args.json_folder:
        if 'dino' in args.embed.lower():
            embed_name = 'dino'
        elif 'clip' in args.embed.lower():
            embed_name = 'clip'
        else:
            embed_name = os.path.basename(args.embed)
        

        for support in args.support:
            for dup in args.dup_thresh:
                if not args.no_concepts:
                    eval_name = 'vcr_k{}_abs{}_area{}_aspect{}_embed_{}_supp{}_dedup{}_rerun1'.format(
                        args.k_concepts,
                        str(int(args.allow_absence)),
                        str(int(args.allow_box_area)),
                        str(int(args.allow_box_aspect)),
                        embed_name,
                        support,
                        dup,
                    )
                else:
                    print('THISSS')
                    eval_name = 'vcr_no_concepts_embed_{}_supp{}_dedup{}_rerun1'.format(embed_name, support, dup)
                    if os.path.exists(os.path.join(args.json_folder, eval_name+'.json')):
                        print(f'Output file {eval_name}.json already exists')
                        continue


                config_str = json.dumps(vars(args))

                eval_path = os.path.join(args.json_folder, eval_name)
                if not args.force and os.path.exists(eval_path):
                    print('no concept at this config already seen')
                    continue
                

                if os.path.isdir(args.json_folder):
                    for f in os.listdir(args.json_folder):
                        if f.endswith('.json') and f != 'args.json':
                            json_path = os.path.join(args.json_folder, f)

                            # Make eval dir in json folder if not exists
                            pathlib.Path(eval_path).mkdir(parents=True, exist_ok=True)
                            out_base = 'vcr_{}'.format(f.replace('.json', ''))


                            # eval_path = os.path.join(args.json_folder, 'eval')
                            out_path = os.path.join(eval_path, out_base)
                            if os.path.exists(out_path+'.json'):
                                print(f'Output file {out_path}.json already exists')
                                continue

                            main(json_path, args.sam, args.img, args.embed, args.fastpath, 
                                args.interact, args.filter, args.padding, args.k_concepts,
                                args.allow_absence, args.allow_box_area, args.allow_box_aspect,
                                args.no_concepts,
                                args.skip_low_err,
                                out_path, config_str, support=support, dedup=dup)
                else:
                    raise ValueError('JSON folder does not exist: {}'.format(args.json_folder))
