Here is a C++ example of k-means on a single GPU, which incidentally shows how the GPU code can be a drop-in replacement for the CPU code:

``` c++
#include <cstdio>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/utils.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

// just generate some random vecs in a hypercube (CPU)
std::vector<float> makeRandomVecs(size_t numVecs, int dim) {
  std::vector<float> vecs(numVecs * dim);
  faiss::float_rand(vecs.data(), vecs.size(), 1);
  return vecs;
}

int main(int argc, char** argv) {
  // Reserves 18% of GPU memory for temporary work by default; the
  // size can be adjusted, or your own implementation of GpuResources
  // can be made to manage memory in a different way.
  faiss::gpu::StandardGpuResources res;

  int dim = 128;
  int numberOfEMIterations = 20;
  size_t numberOfClusters = 20000;
  size_t numVecsToCluster = 5000000;

  // generate a bunch of random vectors; note that this is on the CPU!
  std::vector<float> vecs = makeRandomVecs(numVecsToCluster, dim);
  faiss::gpu::GpuIndexFlatConfig config;
  config.device = 0;            // this is the default
  config.useFloat16 = false;    // this is the default
  faiss::gpu::GpuIndexFlatL2 index(&res, dim, config);

  faiss::ClusteringParameters cp;
  cp.niter = numberOfEMIterations;
  cp.verbose = true; // print out per-iteration stats

  // For spherical k-means, use GpuIndexFlatIP and set cp.spherical = true

  // By default faiss only samples 256 vectors per centroid, in case
  // you are asking for too few centroids for too many vectors.
  // e.g., numberOfClusters = 1000, numVecsToCluster = 1000000 would
  // only sample 256000 vectors.
  //
  // You can override this to use any number of clusters
  // cp.max_points_per_centroid =
  //   ((numVecsToCluster + numberOfClusters - 1) / numberOfClusters);

  faiss::Clustering kMeans(dim, numberOfClusters, cp);

  // do the work!
  kMeans.train(numVecsToCluster, vecs.data(), index);

  // kMeans.centroids contains the resulting cluster centroids (on CPU)
  printf("centroid 3 dim 6 is %f\n", kMeans.centroids[3 * dim + 6]);

  return 0;
}

```

On a P100-PCIe GPU, it generates output that looks like this:
```
Clustering 5000000 points in 128D to 20000 clusters, redo 1 times, 20 iterations
  Preprocessing in 0.585711 s
  Iteration 19 (109.14 s, search 106.78 s): objective=4.51621e+07 imbalance=1.012 nsplit=0       
centroid 3 dim 6 is 0.468013
```

Of note, the actual vectors being clustered above are coming from the CPU. They don't remain resident on the GPU and are only paged in as needed; the memory usage on the GPU is limited. Thus, massive datasets can be clustered.

Part of the k-means code is shared between GPU and CPU (the `faiss::Clustering` object); in fact, our k-means could be made even faster by moving the CPU components to the GPU, but for very large k-means applications (e.g., the clustering for the DEEP1B k-NN graph), the CPU overhead is a small fraction (<5%) of overall work. In the meantime, `faiss::Clustering` requires CPU input and generates CPU output; unlike `GpuIndexFlatL2`, it cannot accept GPU-resident input. Converting the above to CPU k-means requires changing `GpuIndexFlatL2` to `IndexFlatL2`.

Multi-GPU clustering is similar, the `GpuIndexFlatL2` must be replaced with an `IndexProxy` of `GpuIndexFlatL2`s. See [benchs/kmeans_mnist.py](https://github.com/facebookresearch/faiss/blob/master/benchs/kmeans_mnist.py) for an example in Python.

TODO: show multi-GPU clustering in C++