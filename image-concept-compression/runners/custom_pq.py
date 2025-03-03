import faiss
import numpy as np


class CustomProductQuantizer:
    def __init__(self, dim, m, cluster_bits, niter=30):
        self.dim = dim  # Total dimensionality of the vector
        self.m = m  # Number of subquantizers
        self.cluster_bits = cluster_bits  # Bits per subquantizer
        self.n_clusters = 2 ** cluster_bits  # Number of centroids per subquantizer
        self.sub_dim = dim // m  # Dimensionality of each subvector

        # Add these to make compatible with faiss
        self.M = m # number of subquantizers
        self.ksub = self.n_clusters # number of centroids per subquantizer
        self.dsub = self.sub_dim # dimension of each subvector

        # Create m KMeans clusterers
        self.kmeans_clusterers = [
            faiss.Kmeans(self.sub_dim, self.n_clusters, niter=niter, verbose=False)
            for _ in range(self.m)
        ]
        self.centroids = None
        
    def train(self, residuals):
        centroids_list = []
        for i, clusterer in enumerate(self.kmeans_clusterers):
            start = i * self.sub_dim
            end = (i + 1) * self.sub_dim
            sub_vectors = residuals[:, start:end]
            clusterer.train(sub_vectors.astype(np.float32))
            centroids_list.append(clusterer.centroids)
        
        # Store centroids as a numpy array
        self.centroids = np.array(centroids_list)

    # TODO: Double check this
    def reconstruct(self, codes):
        reconstructed = np.zeros((codes.shape[0], self.dim)).astype(np.float32)
        for i in range(self.m):
            start = i * self.sub_dim
            end = (i + 1) * self.sub_dim
            reconstructed[:, start:end] = self.centroids[i][codes[:, i]]
        return reconstructed

    def compute_codes(self, residuals):
        # Adjust dtype according to number of cluster_bits
        dtable = [(8, np.uint8), (16, np.uint16), (32, np.uint32), (64, np.uint64)]
        # codes = np.zeros((residuals.shape[0], self.m), dtype=np.uint8)
        # codes = np.zeros((residuals.shape[0], self.m), dtype=dtable[self.cluster_bits // 8][1])
        dtype_index = 0
        while dtable[dtype_index][0] < self.cluster_bits:
            dtype_index += 1

        codes = np.zeros((residuals.shape[0], self.m), dtype=dtable[dtype_index][1])
        # codes = np.zeros((residuals.shape[0], self.m), dtype=np.uint32)

        for i, clusterer in enumerate(self.kmeans_clusterers):
            start = i * self.sub_dim
            end = (i + 1) * self.sub_dim
            sub_vectors = residuals[:, start:end]
            _, sub_codes = clusterer.index.search(sub_vectors.astype(np.float32), 1)
            codes[:, i] = sub_codes.squeeze()
        return codes

class CustomIVFPQ:
    def __init__(self, dim, n_lists, m, cluster_bits, nprobe=1):
        self.dim = dim
        self.n_lists = n_lists
        self.nprobe = nprobe
        
        # Initialize the coarse quantizer
        self.coarse_quantizer = faiss.IndexFlatL2(dim)
        
        # Initialize the product quantizer
        self.pq = CustomProductQuantizer(dim, m, cluster_bits)
        
        # Initialize the inverted lists
        self.inverted_lists = [[] for _ in range(n_lists)]
        
        # Initialize centroids and codes
        self.centroids = None
        self.codes = None
        self.ids = []  # To store the ids of added vectors
        
        # For efficient reconstruction
        self.id_to_list_and_pos = {}
        
    def train(self, x):
        # Train the coarse quantizer
        kmeans = faiss.Kmeans(self.dim, self.n_lists, niter=20, verbose=False)
        kmeans.train(x.astype(np.float32))
        self.centroids = kmeans.centroids
        self.coarse_quantizer.add(self.centroids)
        
        # Assign training vectors to Voronoi cells
        _, assignments = self.coarse_quantizer.search(x.astype(np.float32), 1)
        assignments = assignments.ravel()
        
        # Compute residuals
        residuals = x - self.centroids[assignments]
        
        # Train the product quantizer on the residuals
        self.pq.train(residuals)
    
    def add(self, x):
        # Quantize vectors to their nearest centroids
        _, assignments = self.coarse_quantizer.search(x.astype(np.float32), 1)
        assignments = assignments.ravel()
        
        # Compute residuals
        residuals = x - self.centroids[assignments]
        
        # Encode residuals using the product quantizer
        codes = self.pq.compute_codes(residuals)
        
        # Add codes to the inverted lists and update id_to_list_and_pos
        for i, (assignment, code) in enumerate(zip(assignments, codes)):
            id = len(self.ids) + i
            pos = len(self.inverted_lists[assignment])
            self.inverted_lists[assignment].append((id, code))
            self.id_to_list_and_pos[id] = (assignment, pos)
        
        # Add ids
        self.ids.extend(range(len(self.ids), len(self.ids) + len(x)))
    
    def reconstruct_n(self, start, end):
        if start < 0 or end > len(self.ids) or start >= end:
            raise ValueError("Invalid start or end index")
        
        reconstructed = np.zeros((end - start, self.dim), dtype=np.float32)
        
        # Collect all the codes and centroid indices for the requested range
        codes = []
        centroid_indices = []
        for i in range(start, end):
            list_id, pos = self.id_to_list_and_pos[i]
            _, code = self.inverted_lists[list_id][pos]
            codes.append(code)
            centroid_indices.append(list_id)
        
        # Convert to numpy arrays for efficient computation
        codes = np.array(codes)
        centroid_indices = np.array(centroid_indices)
        
        # Reconstruct all residuals at once
        residuals = self.pq.reconstruct(codes)
        
        # Add back the centroids
        reconstructed = residuals + self.centroids[centroid_indices]
        
        return reconstructed