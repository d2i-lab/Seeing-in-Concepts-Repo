import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from custom_pq import CustomProductQuantizer

# Make sure to copy the CustomProductQuantizer class implementation here

def test_initialization():
    dim, m, cluster_bits = 128, 8, 8
    pq = CustomProductQuantizer(dim, m, cluster_bits)
    
    assert pq.dim == dim, f"Expected dim to be {dim}, but got {pq.dim}"
    assert pq.m == m, f"Expected m to be {m}, but got {pq.m}"
    assert pq.cluster_bits == cluster_bits, f"Expected cluster_bits to be {cluster_bits}, but got {pq.cluster_bits}"
    assert pq.n_clusters == 2**cluster_bits, f"Expected n_clusters to be {2**cluster_bits}, but got {pq.n_clusters}"
    assert pq.sub_dim == dim // m, f"Expected sub_dim to be {dim // m}, but got {pq.sub_dim}"
    assert len(pq.kmeans_clusterers) == m, f"Expected {m} kmeans clusterers, but got {len(pq.kmeans_clusterers)}"
    
    print("Initialization test passed!")

def test_train_and_compute_codes():
    dim, m, cluster_bits = 128, 8, 8
    pq = CustomProductQuantizer(dim, m, cluster_bits)
    
    np.random.seed(42)
    n_vectors = 100000
    data = np.random.randn(n_vectors, dim).astype(np.float32)
    
    pq.train(data)
    
    subset_size = 10000
    subset_data = data[:subset_size]
    codes = pq.compute_codes(subset_data)
    
    assert codes.shape == (subset_size, m), f"Expected codes shape to be {(subset_size, m)}, but got {codes.shape}"
    # assert codes.dtype == np.uint8, f"Expected codes dtype to be uint8, but got {codes.dtype}"
    assert np.all(codes >= 0) and np.all(codes < 2**cluster_bits), "Codes are out of expected range"
    
    print("Train and compute codes test passed!")

def test_reconstruction_quality():
    dim, m, cluster_bits = 128, 8, 8
    pq = CustomProductQuantizer(dim, m, cluster_bits)
    
    np.random.seed(42)
    n_vectors = 100000
    data = np.random.randn(n_vectors, dim).astype(np.float32)
    
    pq.train(data)
    
    subset_size = 10000
    subset_data = data[:subset_size]
    codes = pq.compute_codes(subset_data)
    
    reconstructed = np.zeros((subset_size, dim), dtype=np.float32)
    for i in range(m):
        start = i * pq.sub_dim
        end = (i + 1) * pq.sub_dim
        centroids = pq.kmeans_clusterers[i].centroids
        reconstructed[:, start:end] = centroids[codes[:, i]]
    
    # Compute similarities between original and reconstructed vectors
    similarities = np.diagonal(cosine_similarity(subset_data, reconstructed))
    
    # Compute similarities between original and random vectors
    random_vectors = np.random.randn(subset_size, dim).astype(np.float32)
    random_similarities = np.diagonal(cosine_similarity(subset_data, random_vectors))
    
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    avg_random_similarity = np.mean(random_similarities)
    min_random_similarity = np.min(random_similarities)
    max_random_similarity = np.max(random_similarities)
    
    print(f"Reconstruction quality metrics:")
    print(f"  Average cosine similarity: {avg_similarity:.4f}")
    print(f"  Minimum cosine similarity: {min_similarity:.4f}")
    print(f"  Maximum cosine similarity: {max_similarity:.4f}")
    print(f"\nRandom vectors similarity metrics:")
    print(f"  Average random cosine similarity: {avg_random_similarity:.4f}")
    print(f"  Minimum random cosine similarity: {min_random_similarity:.4f}")
    print(f"  Maximum random cosine similarity: {max_random_similarity:.4f}")
    
    assert avg_similarity > 0.6, f"Expected average similarity > 0.6, but got {avg_similarity:.4f}"
    assert min_similarity > 0.3, f"Expected minimum similarity > 0.3, but got {min_similarity:.4f}"
    assert avg_similarity > avg_random_similarity, f"Expected average similarity ({avg_similarity:.4f}) to be greater than average random similarity ({avg_random_similarity:.4f})"
    
    print("Reconstruction quality test passed!")

def test_compression_ratio():
    dim, m, cluster_bits = 128, 8, 8
    pq = CustomProductQuantizer(dim, m, cluster_bits)
    
    original_bytes = dim * 4  # 4 bytes per float32
    compressed_bytes = m * 1  # 1 byte per uint8 code
    compression_ratio = original_bytes / compressed_bytes
    
    print(f"Compression ratio: {compression_ratio:.2f}x")
    assert compression_ratio > 1, f"Expected compression ratio > 1, but got {compression_ratio:.2f}"
    
    print("Compression ratio test passed!")

def test_centroid_retrieval():
    dim, m, cluster_bits = 128, 8, 8
    pq = CustomProductQuantizer(dim, m, cluster_bits)
    
    np.random.seed(42)
    n_vectors = 10000
    data = np.random.randn(n_vectors, dim).astype(np.float32)
    
    pq.train(data)
    
    centroids = pq.centroids
    
    expected_shape = (m, 2**cluster_bits, dim // m)
    assert centroids.shape == expected_shape, f"Expected centroids shape {expected_shape}, but got {centroids.shape}"
    
    print("Centroid retrieval test passed!")

# Run the tests
test_initialization()
test_train_and_compute_codes()
test_reconstruction_quality()
test_compression_ratio()
test_centroid_retrieval()