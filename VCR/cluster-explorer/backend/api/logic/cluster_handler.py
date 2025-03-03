import pickle
import numpy as np
from io import BytesIO
import faiss
import os
from collections import defaultdict
from functools import partial
from cuml.manifold import UMAP
from sklearn import preprocessing
import umap
from functools import lru_cache
import time

class ClusterHandler:
    def __init__(self, settings):
        self.seg_embeddings_dir = settings.seg_embeddings_dir
        self.coco_img_dir = settings.coco_img_dir
        self.sam_jsons_dir = settings.sam_jsons_dir
        self.fast_path_dir = settings.fast_path_dir
        self.pretrained_clip_labels_dir = settings.pretrained_clip_labels_dir
        self.umap_embeddings_dir = settings.umap_embeddings_dir
        self.embeddings = None
        self.segment_ids_across_images = None
        self.cluster_labels = None
        self.distances = None

        self.load_embeddings()

    def load_embeddings(self):
        print('Runing get embeddings call')
        fast_path = self.fast_path_dir
        if os.path.exists(fast_path):
            with open(fast_path, 'rb') as f:
                embed_dict = pickle.load(f)

            print('Fast path')
            self.embeddings = embed_dict['average_embeddings']
            self.segment_ids_across_images = embed_dict['segment_ids']
            return

        print('no fast path sir')
        average_embeddings_across_images = []
        segment_ids_across_images = [] 
        imgs = sorted(os.listdir(self.seg_embeddings_dir))

        for idx, seg_emb in enumerate(imgs):
            seg_emb_file = os.path.join(self.seg_embeddings_dir, seg_emb)
            with open(seg_emb_file, "rb") as f:
                dictionary = pickle.load(f)
        
            dictionary["average_embeddings"] = np.load(BytesIO(dictionary["average_embeddings"]))['a']
            average_embeddings = dictionary["average_embeddings"]
            segment_ids = dictionary["segment_ids"]

            if segment_ids[0] == 0:
                average_embeddings = average_embeddings[1:]
                segment_ids = segment_ids[1:]

            average_embeddings_across_images.append(average_embeddings)
            segment_ids_across_images.append(segment_ids)

        average_embeddings_across_images = np.vstack(average_embeddings_across_images)
        
        self.embeddings = average_embeddings_across_images
        self.segment_ids_across_images = segment_ids_across_images



    def reduce_dims(self, reduction_type, embeddings, centroids, target_dimension, faiss_index=None):
        if reduction_type == "pca":
            ncentroids = centroids.shape[0]
            reduced_embeddings = self.apply_pca(np.vstack([embeddings, centroids]), target_dimension)
            reduced_centroids, reduced_embeddings = reduced_embeddings[-ncentroids:], reduced_embeddings[:-ncentroids]
            return reduced_embeddings, reduced_centroids
        elif reduction_type == "cuml_umap":
            return self.apply_cuml_umap(embeddings, target_dimension, centroids)
        elif reduction_type == "umap":
            return self.apply_cuml_umap(embeddings, target_dimension, centroids)
            if faiss_index == None:
                raise Exception()
            else:
                return self.apply_umap(embeddings, target_dimension, faiss_index)


    def apply_pca(self, embeddings, target_dimension):
        original_dimension = embeddings.shape[-1]
        pca = faiss.PCAMatrix(original_dimension, target_dimension)
        pca.train(embeddings)
        return pca.apply(embeddings)
    

        start = time.time()
    def apply_cuml_umap(self, embeddings, target_dimension, centroids):
        umap_n_neighbors = 15
        umap_min_dist = 0
        random_state = 42
        if os.path.exists(self.umap_embeddings_dir):
            print("file exists")
            with open(self.umap_embeddings_dir, 'rb') as f:
                res = pickle.load(f)
                reducer = res["reducer"]
                v_umap = res["v_umap"]
        else:
            print("file does not exist yet")
            # ncentroids = centroids.shape[0]
            vectors = np.vstack(embeddings)
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            print("vectors shape", vectors.shape)
            scaler = preprocessing.StandardScaler().fit(vectors)
            v_scaled = scaler.transform(vectors)

            start = time.time()
            reducer = UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, n_components=target_dimension, random_state=random_state)
            # reducer.fit(v_scaled)
            v_umap = reducer.fit_transform(v_scaled)
            print(f"{time.time() - start} s")
            with open(self.umap_embeddings_dir, 'wb') as f:
                res = {
                    "reducer": reducer,
                    "v_umap": v_umap
                    }
                pickle.dump(res, f)

        vectors = np.vstack(centroids)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        print("vectors shape", vectors.shape)
        scaler = preprocessing.StandardScaler().fit(vectors)
        v_scaled = scaler.transform(vectors)
        centroids_umap = reducer.transform(v_scaled)

        return v_umap, centroids_umap
        

    def apply_umap(self, embeddings, target_dimension, faiss_index):
        n_neighbors = 15
        nn_distances, nn_indices  = faiss_index.search(embeddings, n_neighbors)
        dim_reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, precomputed_knn=(nn_indices, nn_distances), verbose=True, random_state=42)

        # Basically 2d Coords of all embeddings and centroids and nearest points to centroids
        return dim_reducer.fit_transform(embeddings)
    
    def create_index_map(self):
        imgs = sorted(os.listdir(self.seg_embeddings_dir))


        clusters_to_imgsegs = defaultdict(partial(defaultdict, list))
        # idx_to_imgsegs = dict()
        idx_to_imgsegs = list()
        start, index = 0, 0
        for i in range(len(list(os.listdir(self.coco_img_dir)))):
            segment_ids_in_image = self.segment_ids_across_images[i]

            seg_emb = imgs[i]
            img_path = os.path.join(self.coco_img_dir, seg_emb.replace('.pkl', ''))
            img_name = img_path.split('/')[-1].replace('.jpg', '')

            for label, _id in zip(self.cluster_labels[start:start+len(segment_ids_in_image)], segment_ids_in_image):
                clusters_to_imgsegs[int(label)][img_name].append(_id)

            start += len(segment_ids_in_image)

            for segment_id in segment_ids_in_image:
                idx_to_imgsegs.append({
                    "segment_id": segment_id,
                    "img_name": img_name,
                    "cluster_label": int(self.cluster_labels[index])
                })
                # idx_to_imgsegs.append([segment_id, img_name, int(self.cluster_labels[index])])
                index += 1

        return clusters_to_imgsegs, idx_to_imgsegs

    @lru_cache
    def cluster_embeddings(self, ncentroids, dimensionality, embeddingsType):
        print("----- STARTING CLUSTERING -----")

        if dimensionality != -1:
            current_embeddings = self.apply_pca(self.embeddings, dimensionality)
        else:
            current_embeddings = self.embeddings.copy()
        
        kmeans = faiss.Kmeans(current_embeddings.shape[-1], ncentroids, niter=30, verbose=True, gpu=True, seed=42)
        kmeans.train(current_embeddings)

        distances, cluster_ids = kmeans.index.search(current_embeddings, 1)
        cluster_ids = cluster_ids.squeeze()
        self.distances, self.cluster_labels = distances, cluster_ids

        # Create new index to find nearest neighbors to centroids
        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), current_embeddings.shape[-1], faiss.GpuIndexFlatConfig())
        index.add(current_embeddings)
        _, nearest_points_indices = index.search(kmeans.centroids, 5) # Specify how many nearest neighbors to find here

        # Stack centroid embeddings with all embeddings so they get reduced to same space
        # combined_embeddings = np.vstack([current_embeddings, kmeans.centroids])
        # reduced_embeddings = self.reduce_dims("pca", combined_embeddings, 2, faiss_index=index)
        reduced_embeddings, reduced_centroids = self.reduce_dims(reduction_type="cuml_umap", embeddings=current_embeddings, centroids=kmeans.centroids, target_dimension=2, faiss_index=index)
        nearest_points = reduced_embeddings[nearest_points_indices.flatten()].reshape(*(nearest_points_indices.shape), -1)

        centroid_labels = self.label_clusters(kmeans.centroids)

        cluster_map, index_map = self.create_index_map()


        print("----- FINISHED CLUSTERING -----")
        print("----- INFO -----")
        print(reduced_centroids.shape, nearest_points.shape, nearest_points_indices.shape)
        print("----- INFO DONE -----")

        # check this size in bytes of each thing we are returning


        result = {
            "ncentroids": ncentroids,                                   # Number of centroids     
            "centroids": reduced_centroids.tolist(),                    # Coords of centroids
            "nearest_points": nearest_points.tolist(),                  # Coords of nearest points to centroids
            "nearest_points_indices": nearest_points_indices.tolist(),  # Indices of nearest points to centroids
            "cluster_map": cluster_map,                                 # Maps cluster labels -> image name -> list of segment ids
            "index_map": index_map,                                     # Maps index -> image name, segment id, cluster label # Only used for nearest points
            # "cluster_labels": centroid_labels,
            "cluster_names": centroid_labels,

                    # TODO: These should not be hardcoded but rather passed in as arguments
            "input_metadata": {
                "img_dir": self.coco_img_dir,
                "sam_jsons_dir": self.sam_jsons_dir,
                "seg_embeddings_dir": self.seg_embeddings_dir,
        }
        }

        import sys
        for key, value in result.items():
            print(f"Size of {key}: {sys.getsizeof(value)} bytes")

        return result



    def label_clusters(self, embeddings):
        
        with open(self.pretrained_clip_labels_dir, 'rb') as fp:
            d = pickle.load(fp)
            database_embeddings = d["clip_embeddings"]
            words = d["clip_labels"]


        embedding_dim = database_embeddings.shape[1]
        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), embedding_dim, faiss.GpuIndexFlatConfig())
        index.add(database_embeddings)
        label_freq = defaultdict(int)

        labels = dict()
        for i, query_embedding in enumerate(embeddings):
            query_embedding /= np.linalg.norm(query_embedding, keepdims=True)
            k = 1
            distances, indices = index.search(np.expand_dims(query_embedding, 0), k)
            label = words[indices[0][0]]
            if label == "people": label = "person"
            if label_freq[label] == 0:
                labels[i] = label
                label_freq[label] += 1
            else:
                labels[i] = f"{label}_{label_freq[label]}"
                label_freq[label] += 1


        
        return labels

    
    def split_cluster(self, positive_examples, negative_examples, cluster_label):
        # TODO: Further refine this function
        cluster_indices = np.where(self.cluster_labels == cluster_label)
        cluster_indices = cluster_indices[0]

        distances_to_cluster = self.distances[cluster_indices]

        positive_distances = np.mean(np.where(cluster_indices == positive_examples)[0])
        negative_distances = np.mean(np.where(cluster_indices == negative_examples)[0])

        positive_indices = []
        negative_indices = []

        # find the closest indices to the positive and negative examples
        for cluster_index, distance in zip(cluster_indices.tolist(), distances_to_cluster):
            if distance - positive_distances <= distance - negative_distances:
                positive_indices.append(cluster_index)
            else:
                negative_indices.append(cluster_index)

        result = {"positive_indices": positive_indices, 
                "negative_indices": negative_indices}
        
        return result
