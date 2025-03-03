import os
import json
import warnings
import argparse
from collections import Counter
import pathlib

import models
import eval_utils

import numpy as np
from domino import DominoSlicer
from tqdm import tqdm

MODELS = {
    'clip16': 'ViT-B/16',
    'dino_b': 'dinov2_vitb14',
    'dino_l': 'dinov2_vitl14',
}
EVAL_K = [10, 25]

# TODO: Support for multiple gt slices
def run_single_test(n_slices, embeds, logits, gt_slice, y, comp):
    '''
    Given a model, logits, and labels, run a single test and return the
    precision at k.
    '''
    slicer = DominoSlicer(n_slices=n_slices, y_log_likelihood_weight=y, n_mixture_components=min(comp, len(logits)))
    targets = np.ones(len(embeds))
    _ = slicer.fit(embeddings=embeds, targets=targets, pred_probs=logits)
    slicer_probs = slicer.predict_proba(
        embeddings=embeds,
        targets=targets,
        pred_probs=logits
    )

    precisions = {}
    for k in EVAL_K:
        precisions[k] = eval_utils.precision_at_k_single_slice(
            gt_slice, slicer_probs, k
        )
    return precisions

def main(model, json_file: str, img_dir: str, n_slices: int, padding: int, 
         out: str, config_str: str, device:str, y, comp):

    with open(json_file, 'r') as f:
        data = json.load(f)

    if model not in MODELS:
        raise ValueError(f'Invalid model: {model}')

    embeddings = None
    if 'clip' in model:
        print('USING CLIP', MODELS[model])
        embeddings = models.get_embeddings_clip(
            MODELS[model], img_dir, data['img_name'], data['bboxes'],
            padding=padding, device=device
        )
    elif 'dino' in model:
        print('USING DINO')
        embeddings = models.get_embeddings_dino(
            MODELS[model], img_dir, data['img_name'], data['bboxes'],
            padding=padding, device=device,
        )

    precisions = {}
    embeddings = embeddings.cpu().numpy()
    for error_setting in tqdm(data['pred_logits'], desc='Error settings'):
        precisions[error_setting] = {}
        for problem_label in tqdm(data['pred_logits'][error_setting], desc='Problem labels'):
            print('Testing {}, {}'.format(error_setting, problem_label))
            # Create array representing whether each sample is in the slice

            if problem_label.isnumeric():
                gt_slice = np.array(data['cluster_label']) == int(problem_label)
            else:
                gt_slice = np.array(data['cluster_label']) == problem_label
            n_in_slice = np.sum(gt_slice)

            # Precision at k no longer makes sense if there are fewer than k
            # samples in the slice
            if any([n_in_slice < k for k in EVAL_K]):
                warnings.warn(f'Not enough samples in slice {problem_label}')
                continue

            logits = data['pred_logits'][error_setting][problem_label]
            pk = run_single_test(
                n_slices, 
                embeddings,
                logits,
                gt_slice,
                y,
                comp,
            )
            precisions[error_setting][problem_label] = pk

    label_count = Counter(data['cluster_label'])
    precisions['args_dict'] = config_str
    precisions['label_count'] = label_count
    with open(out+'.json', 'w') as f:
        json.dump(precisions, f)
    
    return precisions

def get_img_dir_auto(json_path):
    json_args = json.load(open(json_path, 'r'))['args_dict']
    if isinstance(args, str):
        json_args = json.loads(args)

    if 'img_dir' not in json_args:
        raise ValueError('img_dir not found in args_dict')
    if not os.path.exists(json_args['img_dir']):
        raise ValueError(f'img_dir does not exist: {json_args["img_dir"]}')
    # args.img_dir = json_args['img_dir']
    return json_args['img_dir']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='clip16',
                        choices=MODELS.keys())
    parser.add_argument('--json', '-j', type=str,
        help='JSON file containing img data from coco_cluster_v1.py')
    parser.add_argument('--json_folder', '-jf', type=str,
        help='Folder containing JSON file containing img data from coco_cluster_v1.py')

    parser.add_argument('--img_dir', '-i', type=str, 
                        help='Path to the directory containing the images')
    parser.add_argument('--n_slices', '-n', type=int, default=10,
                        help='Number of slices DOMINO will generate')
    parser.add_argument('--padding', '-p', type=int, default=0,
        help='Padding to add to bounding box (xywh format)')
    parser.add_argument('--out', '-o', type=str,
        help='Output file path for the evaluation data')
    parser.add_argument('--device', '-d', type=str, default='cuda',
        help='Device to run model on')
    parser.add_argument('--force', action='store_true', 
        help='Overwrite output file if it already exists')
    args = parser.parse_args()

    if args.json is not None:
        if args.out is None:
            raise ValueError('Must specify out file when using --json')

        if os.path.exists(args.out):
            raise ValueError(f'Output file already exists: {args.out}')

        if args.img_dir is None:
            args.img_dir = get_img_dir_auto(args.json)
            print('Got img_dir:', args.img_dir)

        config_str = json.dumps(vars(args))
        main(args.model, args.json, args.img_dir, args.n_slices, args.padding, 
            args.out, config_str, args.device)

    if args.json_folder is not None:
        json_files = [f for f in os.listdir(args.json_folder) if f.endswith('.json') and not f.startswith('vcr') and 'args.json' not in f]
        print(json_files)
        for jf in json_files:
            json_path = os.path.join(args.json_folder, jf)
            model_name = args.model

            for y in [40]:
                for comp in [25]:
                    eval_name = 'domino_eval_n{}_embed_{}_y{}_comp{}'.format(args.n_slices, model_name, y, comp)

                    # Make eval dir in json folder if not exists
                    eval_path = os.path.join(args.json_folder, eval_name)
                    pathlib.Path(eval_path).mkdir(parents=True, exist_ok=True)


                    out_base_name = 'domino_{}.json'.format(jf.split('.')[0])
                    out_path = os.path.join(eval_path, out_base_name)

                    if os.path.exists(out_path+'.json') and not args.force:
                        raise ValueError(f'Output file already exists: {out_path}')

                    if os.path.exists(out_path+'.json'):
                        print('i ksip')
                        continue

                    if args.img_dir is None:
                        args.img_dir = get_img_dir_auto(json_path)
                    config_str = json.dumps(vars(args))


                    main(args.model, json_path, args.img_dir, args.n_slices, args.padding, 
                        out_path, config_str, args.device, y, comp)
