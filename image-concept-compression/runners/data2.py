from collections import defaultdict
import bitpacking

from custom_pq import CustomProductQuantizer

from tqdm import tqdm
import numpy as np
import faiss



def get_indices(dim, k_coarse, m, cluster_bits, n_probes, embed_dict,
                use_custom_pq=False, random_seed=None, train_sample_size=None,
                data_sample_size=None):

    embeddings = embed_dict['average_embeddings'] 
    if random_seed is not None:
        np.random.seed(random_seed)

    training_embeds = embeddings
    if train_sample_size is not None:
        print('Training on subset', train_sample_size / embeddings.shape[0])
        sample_indices = np.random.choice(embeddings.shape[0], train_sample_size, replace=False)
        training_embeds = embeddings[sample_indices]

    if data_sample_size is not None:
        print('Utilizing subset', data_sample_size / embeddings.shape[0])
        sample_indices = np.random.choice(embeddings.shape[0], data_sample_size, replace=False)
        embeddings = embeddings[sample_indices]

    # Baseline indexes

    # 1) IVFPQ index on CPU
    print('Building IVFPQ index')
    n_cells = k_coarse
    nbits_per_idx = cluster_bits
    quantizer = faiss.IndexFlatL2(dim)
    index_pq = faiss.IndexIVFPQ(quantizer, dim, n_cells, m, nbits_per_idx)
    index_pq.nprobe = n_probes
    index_pq.train(training_embeds)
    index_pq.add(embeddings)
    # For use in the search function. Must be accounted for in memory usage!
    # index_pq_reconstruct = index_pq.reconstruct_n(0, len(embeddings))

    # 2) Flat index on CPU for original vectors
    print('Building Flat index')
    index_flat_cpu = faiss.IndexFlatL2(dim)
    index_flat_cpu.train(training_embeds)
    index_flat_cpu.add(embeddings)



    # Our custom product quantizer
    # kmeans = faiss.Kmeans(dim, k_coarse, niter=20)
    kmeans = faiss.Kmeans(index_pq.d, index_pq.nlist, niter=index_pq.cp.niter)
    # kmeans.train(training_embeds)
    coarse_centroids = index_pq.quantizer.reconstruct_n(0, index_pq.nlist)
    # kmeans.index.add(coarse_centroids)
    kmeans.train(coarse_centroids)
    print('Kmeans trained (hax)', len(coarse_centroids))
    # d, cluster_ids = kmeans.index.search(embeddings, 1)
    # centroids = kmeans.centroids[cluster_ids].squeeze()
    # residuals = embeddings - centroids
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
    all_images = sorted(embed_dict['img_to_vec_list'])

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

    return index_pq, index_flat_cpu, packd, img_concept_bitmap, all_images, pqq, kmeans