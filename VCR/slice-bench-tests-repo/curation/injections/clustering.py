from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def get_kmeans_clusters(img_list, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=350).fit(img_list)
    return kmeans

def get_hdbscan_clusters(img_list, min_cluster_size=10, method='eom'):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_method=method)
    clusterer.fit(img_list)
    return clusterer

def get_umap_embedding(img_list, n_neighbors=5, min_dist=0.3, n_components=2):
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(img_list)
    return embedding

def plt_umap(embedding, labels):
    # plot non-negative labels
    non_neg = labels >= 0
    plt.figure()
    plt.scatter(embedding[non_neg, 0], embedding[non_neg, 1], c=labels[non_neg], cmap='Spectral', s=5)
    # plot negative labels with alpha
    neg = labels < 0
    plt.scatter(embedding[neg, 0], embedding[neg, 1], c='k', alpha=0.05, s=5)

def get_cluster_quality(X, labels):
    from sklearn.metrics import silhouette_samples
    sample_silhouette_values = silhouette_samples(X, labels)
    return np.mean(sample_silhouette_values)

def get_top_clusters(X, labels, n=2):
    scores = Counter()
    for label in set(labels):
        scores[label] = get_cluster_quality(X, labels == label)
    
    print(scores, scores.most_common(n))
    return scores.most_common(n)

# ------------------------------------------------------------------------------
# Predefined clustering parameters

def kmeans_50(img_list):
    K_CLUSTERS = 50
    return get_kmeans_clusters(img_list, n_clusters=K_CLUSTERS)

def kmeans_20(img_list):
    K_CLUSTERS = 20
    return get_kmeans_clusters(img_list, n_clusters=K_CLUSTERS)

def kmeans_10(img_list):
    K_CLUSTERS = 10
    return get_kmeans_clusters(img_list, n_clusters=K_CLUSTERS)

def umap_hdbscan_001(img_list):
    '''
    Apply UMAP to img_list, then apply HDBSCAN to the embedding with 
    min_cluster_size=0.01 * len(img_list)
    '''
    MIN_SUPP = 0.01
    MIN_DIST = 0.1
    N_NEIGHBORS = 15
    N_COMPONENTS = 10

    min_size = int(MIN_SUPP * len(img_list))
    img_u = get_umap_embedding(img_list, n_neighbors=N_NEIGHBORS, 
                               min_dist=MIN_DIST, n_components=N_COMPONENTS)
    img_u = (img_u - img_u.mean()) / img_u.std()
    return get_hdbscan_clusters(img_u, min_cluster_size=min_size, method='leaf')

def umap_hdbscan_001_eom(img_list):
    '''
    Apply UMAP to img_list, then apply HDBSCAN *w/eom* to the embedding with 
    min_cluster_size=0.01 * len(img_list)
    '''
    MIN_SUPP = 0.01
    MIN_DIST = 0.1
    N_NEIGHBORS = 15
    N_COMPONENTS = 10

    min_size = int(MIN_SUPP * len(img_list))
    img_u = get_umap_embedding(img_list, n_neighbors=N_NEIGHBORS, 
                               min_dist=MIN_DIST, n_components=N_COMPONENTS)
    img_u = (img_u - img_u.mean()) / img_u.std()
    return get_hdbscan_clusters(img_u, min_cluster_size=min_size, method='eom')

def umap_hdbscan_eom_custom(img_list, min_supp=.01, min_dist=.1, n_neighbors=15, n_components=10):
    '''
    Apply UMAP to img_list, then apply HDBSCAN *w/eom* to the embedding with 
    min_cluster_size=0.01 * len(img_list)
    '''
    MIN_SUPP = min_supp
    MIN_DIST = min_dist
    N_NEIGHBORS = n_neighbors
    N_COMPONENTS = n_components

    min_size = int(MIN_SUPP * len(img_list))
    img_u = get_umap_embedding(img_list, n_neighbors=N_NEIGHBORS, 
                               min_dist=MIN_DIST, n_components=N_COMPONENTS)
    img_u = (img_u - img_u.mean()) / img_u.std()
    return get_hdbscan_clusters(img_u, min_cluster_size=min_size, method='eom')

def get_clustering_plugin():
    return {
        'kmeans_50': kmeans_50,
        'kmeans_20': kmeans_20,
        'kmeans_10': kmeans_10,
        'umap_hdbscan_001': umap_hdbscan_001,
        'umap_hdbscan_001_eom': umap_hdbscan_001_eom
    }