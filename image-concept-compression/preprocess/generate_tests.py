from collections import defaultdict
from heapq import heappush, heappop, nlargest
import time
import os
import math
import argparse
import random
import pickle

from tqdm import tqdm
from utils import create_new_pickle, create_grid_visualization

import faiss
import numpy as np
import hashlib

def get_random_segments(embed_dict, n_segments):
    imgs = list(embed_dict['img_to_vec_list'].keys())
    segments = []
    img = None
    while True:
        random_img = random.choice(imgs)
        start_idx, end_idx, _ = embed_dict['img_to_vec_list'][random_img]
        segments_in_img = end_idx - start_idx
        if segments_in_img >= n_segments:
            segments = random.sample(range(start_idx, end_idx), n_segments)
            segments = embed_dict['average_embeddings'][segments]
            img = random_img
            break

    return segments, img

def get_index_filename(func_name, embed_dict):
    # Create a hash of the embed_dict to uniquely identify it
    embed_hash = hashlib.md5(str(embed_dict['average_embeddings'].shape).encode()).hexdigest()[:10]
    return f"{func_name}_{embed_hash}.index"

def save_index(index, filename):
    faiss.write_index(index, filename)

def load_index(filename):
    return faiss.read_index(filename)

def create_index_hsnw_1(embed_dict):
    filename = get_index_filename("hnsw_1", embed_dict)
    if os.path.exists(filename):
        print(f"Loading pre-saved index: {filename}")
        return load_index(filename)

    print(f"Creating new index: {filename}")
    M = 32
    ef_construction = 128
    ef_search = 128
    d = embed_dict['average_embeddings'].shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embed_dict['average_embeddings'])
    save_index(index, filename)
    return index

def create_index_hnsw_2(embed_dict):
    filename = get_index_filename("hnsw_2", embed_dict)
    if os.path.exists(filename):
        print(f"Loading pre-saved index: {filename}")
        return load_index(filename)

    print(f"Creating new index: {filename}")
    M = 32
    d = embed_dict['average_embeddings'].shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.add(embed_dict['average_embeddings'])
    save_index(index, filename)
    return index

def create_index_ivf_flat_1(embed_dict):
    filename = get_index_filename("ivf_flat_1", embed_dict)
    if os.path.exists(filename):
        print(f"Loading pre-saved index: {filename}")
        return load_index(filename)

    print(f"Creating new index: {filename}")
    n_centroids = 32
    d = embed_dict['average_embeddings'].shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_centroids)
    index.nprobe = 2
    embed_subset = int(256 * math.sqrt(len(embed_dict['average_embeddings'])))
    index.train(embed_dict['average_embeddings'][:embed_subset])
    index.add(embed_dict['average_embeddings'])
    save_index(index, filename)
    return index

def create_index_ivf_flat_2(embed_dict):
    filename = get_index_filename("ivf_flat_2", embed_dict)
    if os.path.exists(filename):
        print(f"Loading pre-saved index: {filename}")
        return load_index(filename)

    print(f"Creating new index: {filename}")
    n_centroids = 1024
    d = embed_dict['average_embeddings'].shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_centroids)
    index.nprobe = 10
    embed_subset = int(256 * math.sqrt(len(embed_dict['average_embeddings'])))
    index.train(embed_dict['average_embeddings'][:embed_subset])
    index.add(embed_dict['average_embeddings'])
    save_index(index, filename)
    return index

def create_index_flat(embed_dict):
    filename = get_index_filename("flat", embed_dict)
    if os.path.exists(filename):
        print(f"Loading pre-saved index: {filename}")
        return load_index(filename)

    print(f"Creating new index: {filename}")
    d = embed_dict['average_embeddings'].shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embed_dict['average_embeddings'])
    save_index(index, filename)
    return index

def get_pk_imgs(index, queries, pk, embed_dict):
    # If index is IndexFlatL2, k=entire dataset
    max_radius = len(embed_dict['average_embeddings'])
    radius = pk
    if isinstance(index, faiss.IndexFlatL2):
        radius = len(embed_dict['average_embeddings'])

    all_distances = defaultdict(lambda: defaultdict(list))
    last_expansion = False
    while True:
        radius = min(radius, max_radius)
        distances, indices = index.search(queries, k=radius)
        potential_candidates = []
        all_distances = defaultdict(lambda: defaultdict(list)) 
        # Each query has a number of images associated with it
        # Find corresponding images and store distances to that image for each query
        # This loop essentially adds the vector distance to the [img][query] entry in all_distances
        # But only the closest distance is stored for each image -- hence the set check
        for i in range(len(queries)):
            imgs = [embed_dict['vec_to_img'][idx] for idx in indices[i]]
            potential_candidates.append(set(imgs))
            query_distances = distances[i]
            query_indices = indices[i]
            seen_imgs = set()
            for j, img in enumerate(imgs):
                if img in seen_imgs: # No need to store other distances, since they are further away
                    continue
                seen_imgs.add(img)
                all_distances[img][i].append((query_distances[j], query_indices[j]))

        candidates = set.intersection(*potential_candidates)

        if len(candidates) >= pk or radius >= max_radius:
            if len(candidates) < pk:
                print(f'Warning: Only found {len(candidates)} candidates, expected {pk}')
            if radius >= max_radius:
                print(f'Warning: Reached max radius of {max_radius}')
            break

        radius *= 2

    if len(candidates) < pk:
        print(f'Warning: Only found {len(candidates)} candidates, expected {pk}')
        return None, None

    img_distances = []
    for img in candidates:
        distance = 0
        for j in range(len(queries)):
            dist, _ = all_distances[img][j][0]
            distance += dist
        heappush(img_distances, (-distance, img))

        if len(img_distances) > pk:
            heappop(img_distances)

    result = nlargest(pk, img_distances)
    return [img for _, img in result], [-dist for dist, _ in result]


def main(args):
    if os.path.exists(args.out):
        print(f'Output file {args.out} already exists')
        return

    start_time = time.time()
    embed_dict = create_new_pickle(args.embed_dir, args.pickle)
    print('Embeddings loaded. Building index...')
    index = None
    if args.index_type == "hnsw_1":
        index = create_index_hsnw_1(embed_dict)
    elif args.index_type == "hnsw_2":
        index = create_index_hnsw_2(embed_dict)
    elif args.index_type == "ivf_flat_1":
        index = create_index_ivf_flat_1(embed_dict)
    elif args.index_type == "ivf_flat_2":
        index = create_index_ivf_flat_2(embed_dict)
    elif args.index_type == "flat":
        index = create_index_flat(embed_dict)
    else:
        raise ValueError(f'Index type {args.index_type} not supported')
    end_time = time.time()
    print(f'Time to create index: {end_time - start_time:.2f} seconds')

    if args.debug:
        n_queries = 25
    else:
        n_queries = args.n_queries

    # TODO: Parallelize this.
    visualization_dir = os.path.join(os.path.dirname(args.out), "visualizations")
    os.makedirs(visualization_dir, exist_ok=True)

    unique_imgs = set()
    save_dict = {}
    save_dict['metadata'] = vars(args)
    save_dict['img_ground_truth'] = []
    save_dict['all_queries'] = []

    unique_imgs = set()
    save_interval = max(1, n_queries // 10)  # Save every 10% of progress

    for query_index in tqdm(range(n_queries)):
        segments, img = get_random_segments(embed_dict, args.n_segments)
        unique_imgs.add(img)
        save_dict['all_queries'].append(segments)
        img_gt, dists = get_pk_imgs(index, segments, args.pk, embed_dict)
        save_dict['img_ground_truth'].append((img_gt, dists))

        # Intermittently save progress
        if (query_index + 1) % save_interval == 0 or query_index == n_queries - 1:
            print(f"Saving progress... ({query_index + 1}/{n_queries})")
            with open(args.out, 'wb') as f:
                pickle.dump(save_dict, f)

    print(f'Unique images: {len(unique_imgs)}')
    print(f'Total images: {n_queries}')
    print(f'Percentage of unique images: {len(unique_imgs) / n_queries:.2%}')

    # Final save
    if not args.out.endswith('.pkl'):
        args.out += '.pkl'
    with open(args.out, 'wb') as f:
        pickle.dump(save_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dir", type=str, required=True)
    parser.add_argument("--pickle", type=str)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--index_type", "-i", type=str, required=True)
    parser.add_argument("--n_segments", "-n", type=int, required=True)
    parser.add_argument("--n_queries", "-q", type=int, required=True)
    parser.add_argument("--debug", action="store_true", help="Only run a single iteration w/limited embeddings")
    parser.add_argument("--pk", type=int, help="The number of images needed to be returned from the index")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the original images")
    args = parser.parse_args()
    main(args)
