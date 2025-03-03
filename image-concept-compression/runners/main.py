import os
import math
import yaml
import argparse
import random
from pathlib import Path
import json
import datetime
import itertools

# import data
from search_concept import perform_search
from search_expand import flat_multisearch
from utils import calculate_avg_precision, load_embeddings, create_new_pickle
import mlog
import metrics

import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score

import pickle
from pathlib import Path
from collections import defaultdict

def short_hash(s, length=9):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:length]

def join_constructor(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

yaml.add_constructor('!join', join_constructor)

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def get_index_settings(config, args):
    index_settings = config['index_settings']
    
    # Start with the default setting
    setting = index_settings['default'].copy()
    
    # If an index setting is specified and exists, use it
    if args.index_setting and args.index_setting in index_settings:
        setting = index_settings[args.index_setting].copy()
    elif args.index_setting and (args.index_setting not in index_settings):
        raise ValueError(f"Index setting {args.index_setting} not found in config file.")
    
    # Override with command-line arguments if provided
    overrides = []
    for param in ['k_coarse', 'm', 'nbits', 'nprobes']:
        if getattr(args, param) is not None:
            if setting[param] != getattr(args, param):
                overrides.append(param)
                setting[param] = getattr(args, param)
    
    return setting, overrides


def run_benchmark(args, config, index_setting, active_dataset_config, 
                  logger: mlog.SimpleLogger=None, test_data=None,
                  build_ivf_flat=False, limit=None):
    embed_path = active_dataset_config['embed_path']
    img_path = active_dataset_config['img_path']
    pickle_path = active_dataset_config['pickle_path']

    print('This is pickle path', pickle_path)
    embed_dict = create_new_pickle(embed_path, pickle_path)
    average_embeddings = embed_dict['average_embeddings']
    dim = average_embeddings.shape[1]
    k_coarse = index_setting['k_coarse']
    m = index_setting['m']
    nbits = index_setting['nbits']
    n_probes = index_setting['nprobes']

    logger.info(f"k_coarse: {k_coarse}, m: {m}, nbits: {nbits}, nprobes: {n_probes}")
    logger.info(f"Average embeddings shape: {average_embeddings.shape}")

    train_sample_size = int(128 * math.sqrt(average_embeddings.shape[0]))
    data_sample_size = None

    embed_dict['embed_path'] = embed_path

    # The caching is now handled within get_indices function in data.py
    index_pq, index_flat_cpu, index_ivf_flat_cpu, packd, img_concept_bitmap, all_images, pqq, kmeans = data.get_indices(
        dim=dim, k_coarse=k_coarse, m=m, cluster_bits=nbits, 
        n_probes=n_probes, embed_dict=embed_dict, use_custom_pq=False,
        random_seed=None, train_sample_size=train_sample_size,
        data_sample_size=data_sample_size, build_ivf_flat=build_ivf_flat,
    )

    all_images = list(embed_dict['img_to_vec_list'].keys())
    vec_to_img = embed_dict['vec_to_img']
    
    results = {}
    p_k = 10

    # Add metadata to results
    results['metadata'] = {
        'dataset': args.dataset,
        'test_file': args.test_file or '[None]',
        'k_coarse': k_coarse,
        'm': m,
        'nbits': nbits,
        'nprobes': n_probes,
        'timestamp': datetime.datetime.now().isoformat()
    }

    if test_data:
        n_iterations = len(test_data['all_queries'])
        queries = test_data['all_queries']
        gt_data = test_data['img_ground_truth']
    else:
        n_iterations = 100
        n_segments_list = [1, 2, 4, 6]
        queries = None
        gt_data = None
        raise Exception("No test data provided")
    
    if limit:
        n_iterations = min(n_iterations, limit)

    total_iterations = n_iterations
    with tqdm(total=total_iterations, desc="Benchmark Progress") as pbar:
        all_ap = defaultdict(lambda: {'ours': [], 'pq': [], 'ivf': []})
        all_ar = defaultdict(lambda: {'ours': [], 'pq': [], 'ivf': []})
        counters = defaultdict(lambda: {'ours': [], 'pq': [], 'ivf': []})
        selectivities = defaultdict(lambda: {'ours': [], 'pq': [], 'ivf': []})
        n_imgs = defaultdict(lambda: {'ours': [], 'pq': [], 'ivf': []})
        
        for iteration in range(n_iterations):
            if test_data:
                query = queries[iteration]
                features = {f'v_{i}': query[i] for i in range(len(query))}
                gt_images, gt_distances = gt_data[iteration]
                n_segments = len(query)
            else:
                random_img = random.choice(all_images)
                start_idx, end_idx, _ = embed_dict['img_to_vec_list'][random_img]
                img_embeddings = average_embeddings[start_idx:end_idx]
                n_segments = n_segments_list[iteration % len(n_segments_list)]
                random_embedding_idx = random.sample(range(len(img_embeddings)), min(n_segments, len(img_embeddings)))
                features = {f'v_{i}': img_embeddings[idx] for i, idx in enumerate(random_embedding_idx)}
                gt_images, _, gt_counter = flat_multisearch(index_flat_cpu, features, p_k, vec_to_img, 
                                                max_search_radius=1e6, exclusive_matching=False, logger=logger)
            
            if len(gt_images) < p_k:
                logger.warning(f"Not enough ground truth images for iteration")
                continue

            logger.info(f"-----Begin----- (n_segments: {n_segments})")
            logger.info("--OURS--")
            if not args.disable_ours:
                our_images, _, stat_dict = perform_search(features, p_k, kmeans, pqq, packd, img_concept_bitmap, all_images, 
                                            n_probes=n_probes, exclusive_matching=False,
                                            logger=logger)
                our_counter = stat_dict['counter']
                our_selectivity = stat_dict['estimated_selectivity']
                our_n_imgs = stat_dict['n_imgs']
            else:
                our_images = []
                our_counter = -1
                our_selectivity = -1
                our_n_imgs = -1

            logger.info("--PQ--")
            pq_images, _, stat_dict = flat_multisearch(index_pq, features, p_k, vec_to_img, 
                                            max_search_radius=1e6, exclusive_matching=False,
                                            logger=logger)
            pq_counter = stat_dict['counter']
            pq_selectivity = stat_dict['estimated_selectivity']
            pq_n_imgs = stat_dict['n_imgs']
            logger.info("-----End-----")
            our_ap = metrics._apk(gt_images, our_images, k=p_k)
            pq_ap = metrics._apk(gt_images, pq_images, k=p_k)
            our_ar = metrics._ark(gt_images, our_images, k=p_k)
            pq_ar = metrics._ark(gt_images, pq_images, k=p_k)

            # our_ap = calculate_avg_precision(gt_images, our_images)
            # pq_ap = calculate_avg_precision(gt_images, pq_images)
            our_ap = metrics._apk(gt_images, our_images, k=p_k)
            pq_ap = metrics._apk(gt_images, pq_images, k=p_k)

            all_ap[n_segments]['ours'].append(our_ap)
            all_ap[n_segments]['pq'].append(pq_ap)

            all_ar[n_segments]['ours'].append(our_ar)
            all_ar[n_segments]['pq'].append(pq_ar)

            counters[n_segments]['ours'].append(our_counter)
            counters[n_segments]['pq'].append(pq_counter)

            selectivities[n_segments]['ours'].append(our_selectivity)
            selectivities[n_segments]['pq'].append(pq_selectivity)

            n_imgs[n_segments]['ours'].append(our_n_imgs)
            n_imgs[n_segments]['pq'].append(pq_n_imgs)

            ivf_images = None
            if build_ivf_flat:
                ivf_images, _, stat_dict = flat_multisearch(index_ivf_flat_cpu, features, p_k, vec_to_img, 
                                                max_search_radius=1e6, exclusive_matching=False, logger=logger)
                
                # ivf_ap = calculate_avg_precision(gt_images, ivf_images)
                # ivf_ap = calculate_avg_precision(gt_images, ivf_images)
                ivf_ap = metrics._apk(gt_images, ivf_images, k=p_k)
                ivf_ar = metrics._ark(gt_images, ivf_images, k=p_k)

                ivf_counter = stat_dict['counter']
                ivf_selectivity = stat_dict['estimated_selectivity']
                ivf_n_imgs = stat_dict['n_imgs']

                all_ar[n_segments]['ivf'].append(ivf_ar)
                all_ap[n_segments]['ivf'].append(ivf_ap)
                counters[n_segments]['ivf'].append(ivf_counter)
                selectivities[n_segments]['ivf'].append(ivf_selectivity)
                n_imgs[n_segments]['ivf'].append(ivf_n_imgs)

            pbar.update(1)

    # Calculate final scores for each n_segments
    for n_segments, ap_data in all_ap.items():
        our_ap = np.array(ap_data['ours'])
        pq_ap = np.array(ap_data['pq'])
        our_ar = np.array(all_ar[n_segments]['ours'])
        pq_ar = np.array(all_ar[n_segments]['pq'])

        our_counters = np.array(counters[n_segments]['ours'])
        pq_counters = np.array(counters[n_segments]['pq'])
        our_selectivities = np.array(selectivities[n_segments]['ours'])
        pq_selectivities = np.array(selectivities[n_segments]['pq'])
        our_n_imgs = np.array(n_imgs[n_segments]['ours'])
        pq_n_imgs = np.array(n_imgs[n_segments]['pq'])


        if build_ivf_flat:
            ivf_n_imgs = np.array(n_imgs[n_segments]['ivf'])
            ivf_ap = np.array(ap_data['ivf'])
            ivf_ar = np.array(all_ar[n_segments]['ivf'])
            ivf_counters = np.array(counters[n_segments]['ivf'])
            ivf_selectivities = np.array(selectivities[n_segments]['ivf'])

        results[f'n_segments_{n_segments}_raw'] = {
            'our_ap': our_ap.tolist(),
            'pq_ap': pq_ap.tolist(),
            'our_ar': our_ar.tolist(),
            'pq_ar': pq_ar.tolist(),
            'our_counters': our_counters.tolist(),
            'pq_counters': pq_counters.tolist(),
            'our_selectivities': our_selectivities.tolist(),
            'pq_selectivities': pq_selectivities.tolist(),
            'our_n_imgs': our_n_imgs.tolist(),
            'pq_n_imgs': pq_n_imgs.tolist(),
        }

        if build_ivf_flat:
            results[f'n_segments_{n_segments}_raw']['ivf_ap'] = ivf_ap.tolist()
            results[f'n_segments_{n_segments}_raw']['ivf_ar'] = ivf_ar.tolist()
            results[f'n_segments_{n_segments}_raw']['ivf_counters'] = ivf_counters.tolist()
            results[f'n_segments_{n_segments}_raw']['ivf_selectivities'] = ivf_selectivities.tolist()
            results[f'n_segments_{n_segments}_raw']['ivf_n_imgs'] = ivf_n_imgs.tolist()

        results[f'n_segments_{n_segments}'] = {}
        for key in results[f'n_segments_{n_segments}_raw']:
            results[f'n_segments_{n_segments}'][key] = {
                'mean': float(np.mean(results[f'n_segments_{n_segments}_raw'][key])),
                'std': float(np.std(results[f'n_segments_{n_segments}_raw'][key]))
            }


    return results

def parameter_sweep(args, config, logger: mlog.SimpleLogger=None):
    sweep_settings = config.get('sweep_settings', {})
    if not sweep_settings:
        print("No sweep settings found in config. Running single benchmark.")
        return main(args)

    active_dataset_config = config['datasets'][args.dataset]
    sweep_results = []

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(
        sweep_settings.get('k_coarse', [config['index_settings']['default']['k_coarse']]),
        sweep_settings.get('m', [config['index_settings']['default']['m']]),
        sweep_settings.get('nbits', [config['index_settings']['default']['nbits']]),
        sweep_settings.get('nprobes', [config['index_settings']['default']['nprobes']])
    ))

    # Create output directory for sweep results
    sweep_dir = f"sweep_results_{args.dataset}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(sweep_dir, exist_ok=True)

    for k_coarse, m, nbits, nprobes in tqdm(param_combinations, desc="Parameter Sweep Progress"):
        index_setting = {
            'k_coarse': k_coarse,
            'm': m,
            'nbits': nbits,
            'nprobes': nprobes
        }

        logger.info(f"\nRunning benchmark with settings: {index_setting}")
        results = run_benchmark(args, config, index_setting, active_dataset_config,
                                logger=logger, build_ivf_flat=False)
        sweep_results.append(results)

        # Save individual result
        output_file = f"{sweep_dir}/result_k{k_coarse}_m{m}_nbits{nbits}_nprobes{nprobes}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

    # Save overall sweep results
    overall_output_file = f"{sweep_dir}/sweep_results_summary.json"
    with open(overall_output_file, 'w') as f:
        json.dump(sweep_results, f, indent=4)

    print(f"Sweep results saved in directory: {sweep_dir}")
    return sweep_results

def main(args, logger: mlog.SimpleLogger):
    config = load_config(args.config)
    active_dataset_name = args.dataset
    if active_dataset_name not in config['datasets']:
        logger.error(f"Error: Dataset '{active_dataset_name}' not found in config file.")
        return

    test_data = None
    if args.test_file:
        test_data = load_test_data(args.test_file)

    if args.sweep and 'sweep_settings' in config:
        print("Running parameter sweep...")
        return parameter_sweep(args, config, logger=logger)
    else:
        active_dataset_config = config['datasets'][active_dataset_name]
        index_setting, overrides = get_index_settings(config, args)

        print(f"Active dataset: {active_dataset_name}")
        print(f"\nChosen index setting: {args.index_setting or 'default'}")
        print("Final index settings:")
        for key, value in index_setting.items():
            print(f"{key}: {value}")
        if overrides:
            logger.info("\nOverrides applied:")
            for param in overrides:
                logger.info(f"- {param} was overridden with value: {index_setting[param]}")

        results = run_benchmark(args, config, index_setting, active_dataset_config, 
                                logger=logger, test_data=test_data, build_ivf_flat=args.build_ivf)

        # Save results to JSON file with unique name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'{active_dataset_name}_{args.index_setting or "default"}.json'
        if os.path.exists(output_file):
            output_file = f'{active_dataset_name}_{args.index_setting or "default"}_{timestamp}.json'

        if args.test_file:
            output_file = f'{active_dataset_name}_{args.index_setting or "default"}_{args.test_file.split("/")[-1]}.json'
            if os.path.exists(output_file):
                output_file = f'{active_dataset_name}_{args.index_setting or "default"}_{args.test_file.split("/")[-1]}_{timestamp}.json'


        out_folder = f"{active_dataset_name}/{args.index_setting or 'default'}"
        if os.path.exists(out_folder):
            # Find the next available number
            from glob import glob
            related_folders = glob(f"{out_folder}/{args.index_setting or 'default'}_*")
            if len(related_folders) > 0:
                related_folders_num = [int(folder.split("_")[-1]) for folder in related_folders]
                next_num = max(related_folders_num) + 1
                out_folder = f"{active_dataset_name}/{args.index_setting or 'default'}_{next_num}"
            else:
                out_folder = f"{active_dataset_name}/{args.index_setting or 'default'}_1"

        os.makedirs(out_folder, exist_ok=True)
        output_file = os.path.join(out_folder, output_file)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {output_file}")
        return results

def load_test_data(test_file):
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    return test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with dataset and index configurations")
    parser.add_argument('--config', required=True, help='Path to the configuration file')
    parser.add_argument('--dataset', required=True, help='Name of the dataset to use')
    parser.add_argument('--index_setting', help='Name of the index setting to use')
    parser.add_argument('--k_coarse', type=int, help='Override k_coarse value')
    parser.add_argument('--m', type=int, help='Override m value')
    parser.add_argument('--nbits', type=int, help='Override nbits value')
    parser.add_argument('--nprobes', type=int, help='Override nprobes value')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep if sweep_settings are present in config')
    parser.add_argument('--use_data2', action='store_true', help='Use data2.py instead of data.py')
    parser.add_argument('--test_file', type=str, help='Path to the test file generated by generate_tests.py')
    parser.add_argument('--p_k', type=int, default=10, help='Precision at k setting (default: 10)')
    parser.add_argument('--build_ivf', action='store_true', help='Build IVF-Flat index')
    parser.add_argument('--disable_ours', action='store_true', help='Disable our search')
    parser.add_argument('--limit', type=int, help='Limit the number of tests to run')
    # parser.add_argument('--only_eight', action='store_true', help='Only run on 8 segments')
    args = parser.parse_args()
    if args.use_data2:
        import data2 as data
        print('hi')
    else:
        import data 

    hash_name = vars(args)
    hash_name = ''.join([str(v) for v in hash_name.values()]) + str(datetime.datetime.now())
    hash_name = short_hash(hash_name, length=9)
    logger_name = f"benchmark_{args.dataset}_{hash_name}.log"
    logger = mlog.SimpleLogger(name=logger_name, log_dir="logs", file_logging=True)
    main(args, logger)