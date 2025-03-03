from time import time
from collections import defaultdict
from heapq import heappush, heappushpop, nlargest

import mlog

import faiss
import numpy as np

def compute_distance_table(pq, query):
    """
    Compute the distance table for a given query.
    This is similar to what FAISS does internally.
    """
    m = pq.M  # number of subquantizers
    ksub = pq.ksub  # number of centroids per subquantizer
    dsub = pq.dsub  # dimension of each subvector
    # m parts of size dsub
    query = query.reshape(m, dsub)
    # for each subvector, compute the distance to the centroids
    # distance_table[sub_vector_m, centroid_k] = distance between sub_vector_m and centroid_k
    distance_table = np.zeros((m, ksub), dtype=np.float32)
    # all the centroids, size m * ksub * dsub
    # reorganize the centroids to have m submatrices of size ksub * dsub
    centroids = faiss.vector_to_array(pq.centroids).reshape(m, ksub, dsub)
    # centroids = pq.centroids.reshape(m, ksub, dsub)
    for i in range(m):
        sub_query = query[i]
        sub_centroids = centroids[i]
        # distances (ksub,) = sum((ksub, dsub) - (dsub,) ^ 2, axis=1)
        distances = np.sum((sub_centroids - sub_query) ** 2, axis=1)
        distance_table[i, :] = distances # Distance of subvector i to all centroids

    return distance_table

def adc_distance(distance_table, codes):
    """
    Compute ADC distances using the precomputed distance table and PQ codes.
    """
    # Arrange steps through each "m" subvector
    # code selects the centroid for each subvector
    return np.sum(distance_table[np.arange(len(distance_table)), codes], axis=1)

def reconstruct(pq, codes):
    """
    Reconstruct the original vectors from PQ codes.
    
    Args:
    pq (faiss.ProductQuantizer): The Product Quantizer object
    codes (np.array): PQ codes, shape (n, m) where n is the number of vectors and m is the number of subquantizers
    
    Returns:
    np.array: Reconstructed vectors, shape (n, d) where d is the original vector dimension
    """
    m = pq.M  # number of subquantizers
    ksub = pq.ksub  # number of centroids per subquantizer
    dsub = pq.dsub  # dimension of each subvector
    d = m * dsub  # total dimension of the original vectors
    centroids = faiss.vector_to_array(pq.centroids).reshape(m, ksub, dsub)
    n = codes.shape[0]  # number of vectors to reconstruct
    reconstructed = np.zeros((n, d), dtype=np.float32)
    for i in range(m):
        # Get the codes for this subquantizer
        sub_codes = codes[:, i]
        # Select the corresponding centroids
        sub_centroids = centroids[i, sub_codes]
        # Add to the reconstructed vectors
        reconstructed[:, i*dsub:(i+1)*dsub] = sub_centroids
    
    return reconstructed

# def bitmap_to_pyroaring(bitmap):
#     from pyroaring import BitMap
#     np_bitmaps = [np.where(bitmap)[0] for bitmap in bitmap]
#     roaring_bitmaps = [BitMap(np_bitmap) for np_bitmap in np_bitmaps]
#     return roaring_bitmaps

def improved_bitmap_to_pyroaring(bitmap):
    """
    Convert a NumPy bitmap to PyRoaring bitmaps with careful verification.
    
    Args:
        bitmap: A 2D NumPy boolean array
    
    Returns:
        List of PyRoaring BitMap objects, one for each column
    """
    from pyroaring import BitMap
    import numpy as np
    
    _, n_cols = bitmap.shape
    roarings = []
    
    for col in range(n_cols):
        # Find indices where column is True
        indices = np.where(bitmap[:, col])[0]
        
        # Create BitMap from indices
        # Important: PyRoaring requires uint32 indices
        # roaring = BitMap(indices.astype(np.uint32))
        roaring = BitMap(indices)
        # Verification step
        original_count = np.sum(bitmap[:, col])
        roaring_count = len(roaring)
        assert original_count == roaring_count, f"Column {col} conversion mismatch! Original: {original_count}, Roaring: {roaring_count}"
        
        
        roarings.append(roaring)
    
    return roarings


def bitmap_to_blooms(bitmap, capacity=10_000, err_rate=1/500):
    from rbloom import BloomFilter
    blooms = []
    for col in bitmap.shape[1]:
        indices = np.where(bitmap[:, col])[0]
        blooms.append(BloomFilter(indices))
    return blooms

def bitmap_to_sparse_df(bitmap):
    import pandas as pd
    from scipy import sparse
    # Convert to csr, then to df
    sparse_mat = sparse.csr_matrix(bitmap)
    return pd.DataFrame.sparse.from_spmatrix(sparse_mat)

def perform_search(features, p_k, kmeans, pqq, packd, img_concept_bitmap, all_images, 
                   n_probes=1, exclusive_matching=False,
                   logger: mlog.SimpleLogger = None,
                   bitmap_mode=None,
                   ):
    """
    Perform our quantized search.
    Algorithm steps:
        1. For each feature (query), find the closest centroids for nprobes.
        2. Find the union of the closest centroids for each set of nprobes. (i.e. (dog_1 OR dog_2) AND (cat_1 OR cat_2))
        3. For each image, find distances between the segment embeddings (PQ'd) and our feature queries
        3.1 If exclusive_matching is True, then only allow features to match to single segment embeddings
        3.2 ELSE allow features to match to multiple segment embeddings
        4. For each image, find the closest segment embedding to each feature query
        5. Score images
        6. Return the sorted images

    Arguments:
    features (dict): A dictionary of feature keys to feature vectors
    p_k (int): The number of images to return for use in AP
    kmeans (faiss.IndexFlatL2): The kmeans index
    pqq (faiss.ProductQuantizer): The product quantizer
    packd (dict): The packed PQ codes
    img_concept_bitmap (np.array): A bitmap of images and their concepts
    all_images (list): A list of all image names
    """
    actual_centroids = kmeans.centroids
    closest_centroids = []
    key_to_concept_ids = defaultdict(list)
    counter = 0

    # Track estimated selectivity
    estimated_selectivity = 0
    selected_clusters = set()
    for key, concept_feature in features.items():
        dist, cluster_ids = kmeans.index.search(concept_feature.reshape(1,-1), 1)
        # print('(OURS) Cluster IDs', cluster_ids)
        selected_clusters.add(cluster_ids[0][0])

    selected_clusters = list(selected_clusters)
    valid_imgs = np.all(img_concept_bitmap[:, selected_clusters], axis=1)
    estimated_selectivity = valid_imgs.sum() / len(valid_imgs)
    

    # TODO: This can be done in batches
    for key, concept_feature in features.items():
        dist, cluster_ids = kmeans.index.search(concept_feature.reshape(1,-1), n_probes)
        dist = dist.flatten()
        cluster_ids = cluster_ids.flatten()
        closest_centroids.append((dist, cluster_ids))
        for i, cluster_id in enumerate(cluster_ids):
            key_to_concept_ids[key].append(cluster_id)

    # For each query, union the images that match the closest centroids
    total_nbytes = 0
    if bitmap_mode == None:
        # img_concept_bitmap_df = bitmap_to_sparse_df(img_concept_bitmap)
        # img_concept_bitmap = img_concept_bitmap_df.values
        # img_concept_bitmap = np.array([np.asfortranarray(img_concept_bitmap[:, i]) for i in range(img_concept_bitmap.shape[1])])
        total_nbytes = img_concept_bitmap.nbytes
        centroid_candidates = []
        t1 = time()
        for query_i, (_, cluster_ids) in enumerate(closest_centroids):
            # OR operation within each query's cluster ids
            query_matches = np.any(img_concept_bitmap[:, cluster_ids], axis=1)
            centroid_candidates.append(query_matches)

        # AND operation across all queries
        matching_images = np.where(np.all(centroid_candidates, axis=0))[0]
        t2 = time()
    elif bitmap_mode == 'pyroaring':
        roarings = improved_bitmap_to_pyroaring(img_concept_bitmap)
        total_nbytes = 0
        mem_keys = ['n_bytes_array_containers', 'n_bytes_run_containers', 'n_bytes_bitset_containers']
        for roaring in roarings:
            stats = roaring.get_statistics()
            for key in mem_keys:
                total_nbytes += stats[key]

        centroid_candidates = []
        centroid_candidates_np = []
        t1 = time()
        for query_i, (_, cluster_ids) in enumerate(closest_centroids):

            query_matches = [roarings[cid] for cid in cluster_ids]
            q1 = query_matches[0].copy()
            for q in query_matches[1:]:
                q1 |= q

            centroid_candidates.append(q1)

        q1 = centroid_candidates[0].copy()
        for q in centroid_candidates[1:]:
            q1 &= q

        np_all = np.all(centroid_candidates_np, axis=0)
        np_all = np.sort(np.where(np_all)[0])

        # assert np.array_equal(np_all, list(sorted(q1)))


        matching_images = np.array(list(q1))
        t2 = time()
    elif bitmap_mode == 'blooms':
        import rbloom

        blooms = bitmap_to_blooms(img_concept_bitmap)



    else:
        raise ValueError(f"Invalid bitmap mode: {bitmap_mode}")

    intersection_time = t2 - t1

    # Compute distance table for each [feature_key, concept]
    # If this concept closest to this 
    concept_d_table = {}
    for key, concept_feature in features.items():
        relevant_concept_ids = key_to_concept_ids[key]
        # logger.info("(OURS) #Concept Ids: ", len(relevant_concept_ids))
        for concept_idx in relevant_concept_ids:
            concept_residual = concept_feature - actual_centroids[concept_idx]
            # (feature, centroids)
            concept_d_table[key, concept_idx] = compute_distance_table(pqq, concept_residual)

    # logger.info("(OURS) #Candidate images: ", len(matching_images))
    n_skipped = 0

    # print("Step 5")
    all_distances = {}
    for img_idx in matching_images:
        for key in features:
            for concept_idx in key_to_concept_ids[key]:
                if not img_concept_bitmap[img_idx, concept_idx]:
                    n_skipped += 1
                    continue

                # Calculate distances using ADC
                pq_vectors = np.array(packd[img_idx, concept_idx])
                adc_distances = adc_distance(concept_d_table[key, concept_idx], pq_vectors)
                closest_pq_vector_index = np.argmin(adc_distances)
                closest_pq_vector_distance = adc_distances[closest_pq_vector_index]
                counter += len(adc_distances) # For every distance operation, we need to make this call.

                if (img_idx, key) not in all_distances:
                    all_distances[(img_idx, key)] = closest_pq_vector_distance
                else:
                    if closest_pq_vector_distance < all_distances[(img_idx, key)]:
                        all_distances[(img_idx, key)] = closest_pq_vector_distance

    # logger.info("(OURS) #Skipped: ", n_skipped)

    all_keys = set(features.keys())
    img_distances = []
    for img_idx in matching_images:
        distance = 0
        for key in all_keys:
            if (img_idx, key) in all_distances:
                distance += all_distances[(img_idx, key)]
        
        # Heap uses negative distance to simulate a max heap
        # Furthest images (larger distance -> smaller heap value) will be popped first
        if len(img_distances) < p_k:
            heappush(img_distances, (-distance, img_idx))
        elif -distance > img_distances[0][0]:
            # Heap is full and we are closer than the furthest image
            heappushpop(img_distances, (-distance, img_idx))
        
    
    stat_dict = {
        'counter': counter,
        'estimated_selectivity': estimated_selectivity,
        'n_imgs': len(matching_images)
    }
    image_distances_named = {img_idx: -dist for dist, img_idx in img_distances}
    sorted_images = [all_images[idx] for _, idx in nlargest(p_k, img_distances)]

    # return sorted_images, image_distances_named, debug_reconstruction

    return sorted_images, image_distances_named, stat_dict, intersection_time, total_nbytes