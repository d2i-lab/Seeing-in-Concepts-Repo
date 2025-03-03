from collections import defaultdict

import utils
import mlog


import numpy as np
import faiss
from scipy.optimize import linear_sum_assignment

# def euclidean_distance(v1, v2):
#     return np.linalg.norm(v1 - v2, ord=2)

def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.sum((v1 - v2) ** 2)
    
def flat_multisearch(search_index, features, p_k, 
                     vec_to_img,
                     max_search_radius=1e5, exclusive_matching=False,
                     logger: mlog.SimpleLogger = None
                     ):
    '''
    Perform multi-vector flat search.

    Args:
    search_index: Index to search on.
    features (dict): Dictionary of features to search for. [Key: np.array]
    embed_source (np.array): Source of embeddings to search on.
    p_k (int): Precision at k setting.
    vec_to_img (list): Mapping of vector index to image index.
    img_to_vec_list (dict): Mapping of image index to start and end index of vectors.
    max_search_radius (float): Maximum search radius for the search.
    exclusive_matching (bool): If True, features cannot map to same concept in 
        image. Use Hungarian matching to minimize distance.
    '''
    do_log = False
    search_radius = p_k
    if hasattr(search_index, 'invlists'):
        invlists = search_index.invlists
        max_search_radius =  max([invlists.list_size(l) for l in range(invlists.nlist)]) * search_index.nprobe
        # search_radius = max_search_radius
        search_radius = p_k * 100
        do_log = True

        logger.info("(PQ): Index settings")
        logger.info(f"(PQ): #Clusters: {search_index.nlist}")
        # logger.info("(PQ): #Probe: ", search_index.nprobe)


    # Track number of points compared to
    counter = 0

    candidates = set()
    all_distances = defaultdict(lambda: defaultdict(list))
    while True:
        potential_candidates = []
        temp_distances = defaultdict(lambda: defaultdict(list))
        # Prepare batch of features for search
        feature_batch = np.array(list(features.values()))
        distances, indices = search_index.search(feature_batch, search_radius)
        if isinstance(search_index, faiss.IndexIVFPQ) or isinstance(search_index, faiss.IndexIVFFlat):
            invlist_info = utils.get_query_invlists_info(search_index, feature_batch)
            for invlist_id, invlist_size in invlist_info:
                counter += invlist_size



        # indices should be of size (n_features, search_radius)
        for i, (key, feature) in enumerate(features.items()):
            img_names = [vec_to_img[idx] for idx in indices[i]]
            potential_candidates.append(set(img_names))

            # Store distances for each image
            for j, img in enumerate(img_names):
                temp_distances[img][key].append((distances[i][j], indices[i][j]))


        candidates = set.intersection(*potential_candidates)
        search_radius *= 2
        if do_log:
            logger.info("(PQ): Expanding search radius to ", search_radius)
        # Break if we exceed the maximum search radius
        if (search_radius > max_search_radius) or (len(candidates) >= p_k):
            for key in features:
                for img in candidates:
                    all_distances[img][key] = temp_distances[img][key]
            break

    if do_log:
        logger.info("(PQ): #Candidates: ", len(candidates))

    if len(candidates) < p_k:
        logger.warning(f"(PQ): Only found {len(candidates)} candidates, returning all")

    # Score each set of candidates
    img_scores = {}
    for img in all_distances:
        distances_idx = all_distances[img]
        if exclusive_matching:
            all_possible_idxs = set()
            for key in distances_idx:
                all_possible_idxs.update([idx for _, idx in distances_idx[key]])
            
            # Create cost matrix
            unassigned_penalty = 1e9
            cost_matrix = np.full((len(features), len(all_possible_idxs)), unassigned_penalty)
            idx_to_col = {idx: j for j, idx in enumerate(all_possible_idxs)}
            
            for i, key in enumerate(features):
                for dist, vec_idx in distances_idx[key]:
                    j = idx_to_col[vec_idx]
                    cost_matrix[i, j] = dist

            # Perform Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()
            # Count number of unassigned features
            unassigned_features = len(features) - len(row_ind)
            # Final score is the total cost plus penalties for unassigned features
            img_scores[img] = total_cost + unassigned_features * unassigned_penalty
        else:
            # Default to summing all distances
            img_scores[img] = 0
            distances_idx = all_distances[img]
            for key in distances_idx:
                img_scores[img] += distances_idx[key][0][0]

    stat_dict = {
        'counter': counter,
        'estimated_selectivity': -1,
        'n_imgs': len(candidates)
    }



    sorted_images = sorted(img_scores.items(), key=lambda x: x[1])
    sorted_images = [img for img, _ in sorted_images[:p_k]]
    return sorted_images[:p_k], sorted_images, stat_dict
