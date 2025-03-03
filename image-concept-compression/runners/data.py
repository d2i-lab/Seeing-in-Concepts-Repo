from collections import defaultdict
import bitpacking
import pickle

from custom_pq import CustomProductQuantizer

from tqdm import tqdm
import numpy as np
import faiss
import os
import hashlib

def get_cache_folder(embed_path, k_coarse, m, cluster_bits, n_probes, train_sample_size, data_sample_size):
    params = f"{embed_path}_{k_coarse}_{m}_{cluster_bits}_{n_probes}_{train_sample_size}_{data_sample_size}"
    params_hash = hashlib.md5(params.encode()).hexdigest()[:10]
    return f"cached_index_{params_hash}"

def save_index_data(cache_folder, index_pq, coarse_centroids, train_indices, all_images):
    os.makedirs(cache_folder, exist_ok=True)
    faiss.write_index(index_pq, os.path.join(cache_folder, "index_pq.faiss"))
    np.save(os.path.join(cache_folder, "coarse_centroids.npy"), coarse_centroids)
    np.save(os.path.join(cache_folder, "train_indices.npy"), train_indices)
    with open(os.path.join(cache_folder, "all_images.pkl"), "wb") as f:
        pickle.dump(all_images, f)

def load_index_data(cache_folder):
    index_pq = faiss.read_index(os.path.join(cache_folder, "index_pq.faiss"))
    coarse_centroids = np.load(os.path.join(cache_folder, "coarse_centroids.npy"))
    train_indices = np.load(os.path.join(cache_folder, "train_indices.npy"))
    with open(os.path.join(cache_folder, "all_images.pkl"), "rb") as f:
        all_images = pickle.load(f)
    return index_pq, coarse_centroids, train_indices, all_images

def hex_string_to_int(hex_string):
    return int(hex_string, 16)


def get_indices(dim, k_coarse, m, cluster_bits, n_probes, embed_dict,
                use_custom_pq=False, random_seed=None, train_sample_size=None,
                data_sample_size=None, build_ivf_flat=False, cache_enabled=False):

    
    dataset_hash = hashlib.md5(embed_dict['embed_path'].encode()).hexdigest()[:7]
    np.random.seed(int(dataset_hash, 16))
    cache_folder = get_cache_folder(embed_dict['embed_path'], k_coarse, m, cluster_bits, n_probes, train_sample_size, data_sample_size)
    embeddings = embed_dict['average_embeddings'] 


    # Check if cached index exists
    training_embeds = None
    if cache_enabled and os.path.exists(os.path.join(cache_folder, "index_pq.faiss")):
        print(f"Loading cached index from {cache_folder}")
        index_pq, coarse_centroids, train_indices, all_images = load_index_data(cache_folder)
        training_embeds = embeddings[train_indices]
    else:
        print("Building new index")
        
        all_images = sorted(embed_dict['img_to_vec_list'])
        # if random_seed is not None:
        #     np.random.seed(random_seed)

        train_indices = np.arange(embeddings.shape[0])
        if train_sample_size is not None:
            print('Training on subset', train_sample_size / embeddings.shape[0])
            train_indices = np.random.choice(embeddings.shape[0], train_sample_size, replace=False)
        training_embeds = embeddings[train_indices]

        if data_sample_size is not None:
            print('Utilizing subset', data_sample_size / embeddings.shape[0])
            sample_indices = np.random.choice(embeddings.shape[0], data_sample_size, replace=False)
            embeddings = embeddings[sample_indices]
            new_all_images = set()
            for idx in sample_indices:
                new_all_images.add(embed_dict['vec_to_img'][idx])

            all_images = list(sorted(new_all_images))

        # Build IVFPQ index
        print('Building IVFPQ index')
        n_cells = k_coarse
        nbits_per_idx = cluster_bits
        quantizer = faiss.IndexFlatL2(dim)
        index_pq = faiss.IndexIVFPQ(quantizer, dim, n_cells, m, nbits_per_idx)
        index_pq.nprobe = n_probes
        index_pq.train(training_embeds)
        index_pq.add(embeddings)

        # Get coarse centroids
        coarse_centroids = index_pq.quantizer.reconstruct_n(0, index_pq.nlist)

        # Save the newly created index data
        save_index_data(cache_folder, index_pq, coarse_centroids, train_indices, all_images)
        print(f"Saved new index to {cache_folder}")

    # Reconstruct kmeans from coarse centroids
    kmeans = faiss.Kmeans(index_pq.d, index_pq.nlist, niter=index_pq.cp.niter)
    kmeans.train(coarse_centroids)
    print('Kmeans trained (hax)', len(coarse_centroids))

    # Build flat index (not cached)
    print('Building Flat index')
    index_flat_cpu = faiss.IndexFlatL2(dim)
    # index_flat_cpu.add(embed_dict['average_embeddings'])
    embeddings = embed_dict['average_embeddings']
    # Our custom product quantizer
    # kmeans = faiss.Kmeans(dim, k_coarse, niter=20)
    # kmeans.train(training_embeds)
    # coarse_centroids = index_pq.quantizer.reconstruct_n(0, index_pq.nlist)

    if not build_ivf_flat:
        index_flat_cpu.add(embeddings)

    index_ivf_flat_cpu = None
    if build_ivf_flat:
        print('Building IVF-Flat index')
        index_ivf_flat_cpu = faiss.IndexIVFFlat(index_flat_cpu, dim, index_pq.nlist)
        index_ivf_flat_cpu.nprobe = n_probes
        index_ivf_flat_cpu.train(coarse_centroids)
        index_ivf_flat_cpu.add(embeddings)

    # kmeans.index.add(coarse_centroids)
    # kmeans.train(coarse_centroids)
    # d, cluster_ids = kmeans.index.search(embeddings, 1)
    # centroids = kmeans.centroids[cluster_ids].squeeze()
    # residuals = embeddings - centroids

    print("This is training embeds", training_embeds.shape)
    d, cluster_ids = kmeans.index.search(training_embeds, 1)
    centroids = kmeans.centroids[cluster_ids].squeeze()
    train_residuals = training_embeds - centroids

    pqq = None
    if use_custom_pq:
        pqq = CustomProductQuantizer(dim, m, cluster_bits)
    else:
        pqq = faiss.ProductQuantizer(dim, m, cluster_bits)
    # pqq.train(residuals)
    # codes = pqq.compute_codes(residuals)
    pqq.train(train_residuals)

    d, cluster_ids = kmeans.index.search(embeddings, 1)
    centroids = kmeans.centroids[cluster_ids].squeeze()
    codes = pqq.compute_codes(embeddings - centroids)

    packd = bitpacking.NumpyBitpackedDict()
    # packd = {}
    n_imgs = len(embed_dict['img_to_vec_list'])
    img_concept_bitmap = np.zeros((n_imgs, k_coarse), dtype=bool)

    for img_idx, img_names in tqdm(enumerate(all_images)):
        start, end, segment_ids_idx = embed_dict['img_to_vec_list'][img_names]
        cluster_assignments = cluster_ids[start:end].flatten() # Cluster assignments
        residual_pq = codes[start:end]

        for concept_feature, pq in zip(cluster_assignments, residual_pq):
            img_concept_bitmap[img_idx, concept_feature] = True
            if (img_idx, concept_feature) not in packd:
                packd[img_idx, concept_feature] = [pq]
            else:
                packd[img_idx, concept_feature].append(pq)


    #3) IVFOPQ index on CPU
    # print('Building IVFOPQ index')

    return index_pq, index_flat_cpu, index_ivf_flat_cpu, packd, img_concept_bitmap, all_images, pqq, kmeans
