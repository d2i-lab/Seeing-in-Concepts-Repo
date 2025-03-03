
# Math/algorithmic questions

## How can I get constructive criticism about my data?

To analyze a matrix, print

`MatrixStats(my_matrix).comments` (Python)

`MatrixStats(n, d, my_matrix).comments` (C++)

this will output some (hopefully readable) comments on the matrix content: are there NaNs? Duplicate vectors? Constant dimensions? Are the vectors normalized? It is always useful to run this when faced with some weird behavior in Faiss.

## Why do I get weird results with brute force search on vectors with large components?

Keep in mind that floating-point computations are prone to round-off errors. These are particularly visible when floats of different magnitudes are added ("catastrophic cancellation"). 

Two examples: 

- if components of a vector are large and slightly different, and the decomposition `||x-y||^2 = ||x||^2 + ||y||^2 - 2 * <x, y>` is used (this is the case with `IndexFlat` with batches of > 20 query vectors, see [here](https://github.com/facebookresearch/faiss/wiki/Implementation-notes#matrix-multiplication-to-do-many-l2-distance-computations)), then the differences are may be cancelled out, see [this example](https://gist.github.com/mdouze/efc94c57e2302469287b9d1a2501d277). A workaround is to center the vector components, which does not change the distances but improves the problem's conditioning. 

- if some components are much larger than others, then during accumulation of distances, smaller components may be cancelled out, see [this example](https://gist.github.com/mdouze/a1ecc330a7c0685e25916bcb526007f1). Here there is no real workaround. Fortunately it happens mostly with vectors that are far apart, and thus are hopefully not relevant for similarity search.

A formalization of finite precision computations: see chapter 2.7 in the book "Matrix computations", Golub & van Loan, Hopkins univ press.

## Analyzing accuracy issues with `IndexIVFPQ`

`IndexIVFPQ` (aka `"IVFx,PQy"`) relies on vector compression and an inverted list that restricts the distance computations to just a fraction of the dataset. 
If the accuracy of an `IndexIVFPQ` is too low:

- set the `nprobe` to the number of centroids to scan the whole dataset, and see how it performs. The accuracy loss at that point is due to just the PQ compression. Note that the default `nprobe` is 1, which is on the low side. 

- build `IndexIVFFlat` (aka `"IVFx,Flat"`) instead of `IndexIVFPQ`. This will show how much accuracy is lost due to the non-exhaustive search. 

The combination of both should yield 100% accuracy.
If the accuracy is not 100%, this could be due to ties in the distances, ie. the ordering of results is arbitrary.

## Why is multi-level IVF search not supported in Faiss?

The IVFADC and other IVFxx indexing methods can be seen as a special case of a tree-based search with only 2 levels and large leaves. 

The reason why leaves are so large is because it is efficient to perform linear scans in memory, especially in the product quantization case where distance computations can be factorized and stored in precomputed tables. 

Extending to more than 2 levels is tricky because the most accurate way of finding which leaves to visit is to compute distances to all leave centroids, which is what IVFADC does. Adding more branching levels would speed up the search but will decrease this accuracy. The paper "Scalable recognition with a vocabulary tree”, Nister & Stewenius, CVPR’06 does this.

Another type of methods is to use a graph between points and to a BFS on this graph to find the nearest neighbors (HNSW method and variants). This is currently the recommended way to select the leaves to visit in Faiss. This can be seen as a quantization method. 

3-level indexes have been used in "Searching in one billion vectors: re-rank with source coding”, Jegou & al., ICASSP’11 and is implemented as IVFPQR in Faiss. It does provide some improvement over IVFADC in terms of accuracy but is quite hard to tune. 

From a pure encoding perspective, the most accurate methods do use non-orthogonal codebooks, i.e. sums of arbitrary vectors. However encoding is slow. See [Additive quantizers](https://github.com/facebookresearch/faiss/wiki/Additive-quantizers) on how to use that.

## How can I reproduce results from the original PQ paper?

The paper in question is ['Product quantization for nearest neighbor search'](https://hal.inria.fr/inria-00514462/document), Jegou, Douze, Schmid, PAMI'11. 

See [#1045](https://github.com/facebookresearch/faiss/issues/1045)

## How can I index 2D or 3D data?

Indexing low-dimensional data (for any dimension) is not addressed well in Faiss. 
This is because these cases are better addressed with tree-based structures like kd-trees: they offer exact search results at logarithmic search time. 
An example implementation is in [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html).

See also: [#1709](https://github.com/facebookresearch/faiss/issues/1709) [#1630](https://github.com/facebookresearch/faiss/issues/1630)


# Questions on the API 

## Can searches and additions be performed concurrently from several threads?

Concurrent searches are supported for the CPU code, but not the GPU code. Concurrent search/add or add/add are not supported. There is no locking mechanism in place to protect against this, so the calling code should maintain a lock. 

See #492 for workarounds. 

## Why are searches with a single query vector slow?

Faiss is optimized for batch search. There are three reasons for that: 

- most indexes rely on a clustering of the data that at query time requires a matrix-vector multiplication (for a single query vector) or matrix-matrix multiplication (for a batch of queries). Matrix-matrix multiplications are often much faster than the corresponding amount of matrix-vector multiplications. 

- search parallelization is over the queries. Doing otherwise would require to maintain several result lists per thread and merging them on output, a source of overhead.

- in a multithreaded environment, several searches can be performed concurrently, to fully occupy the processing cores of the machine.


## In C++, do I need to keep a reference to the coarse quantizer around for an IndexIVFFlat/IndexIVFPQ/IndexIVFPQR index?

If you construct the coarse quantizer yourself, the code assumes by default that you will delete it. To transfer ownership to the `IndexIVF`, set `own_fields` to true. 

If you constructed the index with the `index_factory`, `read_index` or `clone_index` then all sub-indexes belong to the object returned by the function, so there is no need to worry about ownership.

In Python, ownership management is automatic. See also: https://github.com/facebookresearch/faiss/wiki/Troubleshooting#crashes-in-pure-python-code

## How can I distribute index building on several machines?

For this you can: 

1. train an IndexIVF* on a representative sample of the data, store it.

2. for each node, load the trained index, add the local data to it, store the resulting populated index

3. on a central node, load all the populated indexes and merge them. Here is a C++ example on how to merge: [test_merge.cpp](https://github.com/facebookresearch/faiss/blob/master/tests/test_merge.cpp)

If the data on the different machines has a different distribution, then it may be beneficial to do a separate training on each of the machines and merge the results at search time:

1. on each node, train an index on this node's data + add the data to the index

2. save the node's index

3. load all the indexes on a central machine and combine them into an `IndexShards`.

Note that if the index uses strong compression, this second solution may yield distances that are hard to compare, and thus the overall indexing accuracy may be worse than doing a common training. YMMV.

## Does Faiss support string ids?

Faiss does not support string ids for vectors (or any datatype other than 64-bit ints). 
It is unlikely that this will change. 
See [issue #641](https://github.com/facebookresearch/faiss/issues/641) for a discussion of this topic.

<!--
## Why does python Faiss not accept `float64`/`float32`/`float16` vectors or `uint64` ids or non-contiguous arrays?

**Update:** automatic conversion is supported in Faiss 1.7.3

Python Faiss is intended to be a shallow wrapper above C++. 
Therefore the choice was to not implicitly convert data when the C++ API does not support them. 

Then why does the C++ layer not support double precision? 
Because Faiss is a performance-oriented library and double precision is slower with almost no benefit in precision. 
--> 

## Searching duplicate vectors is slow

The `IndexIVF` and `IndexHSNW` variants have trouble indexing large numbers of identical vectors. 
In the `IndexIVF` case this is because they all end up in the same inverted list, that must then be scanned sequentially. 
See #1097 for an analysis of the `IndexHNSW` case. 
This also applies to near-duplicate vectors.

The workaround to this is to de-duplicate vectors prior to indexing. 
Faiss does not do that by default because it would have a run-time and memory impact for use cases where there are no duplicates. 
However, the `IndexFlatDedup` index does de-duplication. 
Also, `MatrixStats` will find whether a dataset has duplicates.

## Reproducibility 

In Faiss, all random seeds are set to constant values, which normally means that two runs of training produce bit-exact same results. 
This is not always possible for performance reasons, see [Reproducibility with multiple threads](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls#reproducibility-with-multiple-threads).

Explicit seeds can be provided to the random generators, see for example  [demo_seeds_ivfpq](https://gist.github.com/mdouze/1892178b5663b80e85ab076966c59c28).

# Questions about training 

## Can I ignore "WARNING clustering XXX points to YYY centroids"?

When applying k-means algorithm to cluster n points to k centroids, there are several use cases: 

- n < k: this raises an exception with an assertion because we cannot do anything meaningful

- n < min_points_per_centroid * k: this produces the warning above. It means that usually there are too few points to reliably estimate the centroids. This may still be ok if the dataset to index is as small as the training set.

- n < max_points_per_centroid * k: comfort zone

- n > max_points_per_centroid * k: there are too many points, making k-means unnecessarily slow. Then the training set is sampled. 

The parameters {min,max}_points_per_centroids (39 and 256 by default) belong to the [`ClusteringParameters`](https://github.com/facebookresearch/faiss/blob/master/faiss/Clustering.h) structure. 
They can be changed to quiet the warnings, eg setting `cp.min_points_per_centroid = 1` and `cp.max_points_per_centroid = 10000000`. 

In the python `KMeans` object the fields can be set as parameters to the constructor.
For indexes, the k-means routine is called for `IndexPQ`, `IndexIVFFlat`, and twice for `IndexIVFPQ` (once for the coarse quantizer with k=`ncentroids`, once for the PQ, with k=2^`nbits_per_idx`), and three times for `IndexIVFPQR`. 
In that case, the appropriate `ClusteringIndex` is accessible as `IndexPQ::pq::cp` and `IndexIVF::cp` (`index.pq.cp` and `index.cp` in Python).

## How many training points do I need for k-means?

As a rule of thumb there is no consistent improvement of the k-means quantizer beyond 20 iterations and 1000 * k training points. And these number diminish when k increases (ie. when training for larger k you can do fewer iterations on fewer training points). This is why the k-means clustering function samples vectors by default (see previous question).

As an example, the results you'd see from clustering 6.2B vectors to 80K centroids would probably be about the same quality-wise as sampling a random 20.48M subset to 80K centroids, so it's just saving you work (of course sampling should be unbiased, see next question). 


## How can I do unbiased sampling from a large set of vectors?

If the set fits in RAM, the answer is easy, just sample elements without replacement.

In Python
```python
rs = np.random.RandomState(123)
idx = rs.choice(nt_sample, size=nt, replace=False)
xt = xt[idx]
```

In C++ there is a helper function: [`fvecs_maybe_subsample`](https://github.com/facebookresearch/faiss/blob/36ddba9196f19b640d5ba2ead558d50e02ecde89/utils/utils.h#L150).

If the dataset does not fit in RAM and/or comes from a stream, you need [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling). 
Here is an example implementation in Python: [reservoir_sampling.ipynb](https://gist.github.com/mdouze/92c5bafcf2b91356cf5e799e3889a0e9)

## Can I use k-means with an inner-product index? 

Running k-means with an inner-product dataset is supported. 
The assignment then uses maximum inner product search and the centroids are updates with the mean of the vectors assigned to it. 

Note however that k-means originally aims at minimizing the squared Euclidean distance between the training vectors and the centroids they are assigned to. The convergence properties of k-means are guaranteed in this setting. When using other metrics like inner product, there is no explicit loss to optimize and convergence is not guaranteed. 

See also: [#2363](https://github.com/facebookresearch/faiss/issues/2363) [#2466](https://github.com/facebookresearch/faiss/issues/2466)


## Can I do k-means with something else than L2 assignment? 

This is computationally possible. However, the convergence guarantees and loss minimization properties hold only for the squared L2 loss. 

One typical case is training quantizers used for inner product assignment. 
Empirically, it turns out that it is better to renormalize the centroids after each iteration (set the `spherical` field of the C++ `Clustering` or the Python `Kmeans` object to true).

See also discussion in [issue #2363](https://github.com/facebookresearch/faiss/issues/2363)
And section 5.1 in the [Faiss paper](https://arxiv.org/pdf/2401.08281).

## How can I warm-start the centroid training for an IVF index?

To adapt eg. to a drifting data distrubution, see how to warm-start the training here: [warm_start_ivf_centroids.ipynb](https://gist.github.com/mdouze/fd0e3e1f20a3d530d96b0e017050df43#file-warm_start_ivf_centroids-ipynb)

You may be interested in the paper [DEDRIFT: Robust Similarity Search under Content Drift](https://openaccess.thecvf.com/content/ICCV2023/html/Baranchuk_DEDRIFT_Robust_Similarity_Search_under_Content_Drift_ICCV_2023_paper.html)


## Can I used an index trained on some kind of data to index some other type of data of the same dimension?

No.

The objective of the training stage is to exploit the distribution of the data (clusters, sub-spaces) to improve the efficiency of the index. The distribution is estimated on a sample provided at train time, that should be representative of the data that is indexed. This is of course the case when the train set is the same as the added vectors.

When adding data and searching, Faiss checks only whether the dimensionality of the data is correct (and this only in the Python wrappers). If the distribution is incorrect, this will result in degraded performance in terms of accuracy and/or search time. 

Cases when a new training is required: 

- when re-training a CNN that produces descriptors that are indexed

- when the type of media you index becomes statistically different (eg. class1 grows from 1% to 90% of the data)

## Is re-training an index supported?

No it is not. The states for the index are: 

1. `is_trained = false, ntotal = 0`, transition to 2 with `train()`

2. `is_trained = true, ntotal = 0`, transition to 3 with `add()`

3. `is_trained = true, ntotal > 0`, transition to 2 with `reset()`

Since a new index is just a bunch of parameters, it is not worthwhile to support "un-training" an index, it is simpler to just construct a new one.

# Extracting information from a C++ Index object

[Here](https://github.com/facebookresearch/faiss/wiki/Python-C---code-snippets) we give some handy code in Python notebooks that can be copy/pasted to perform some useful operations.

# Other questions 

## Why do I get "wrong number or type of arguments" errors in Python? 

This often happens when a numpy int is passed in to a C++ function that expects an int.
For example: 
```python
import faiss
import numpy as np

ncent = np.int32(123)
clus = faiss.Clustering(10, ncent)
```
fails. The workaround is to explicitly cast the value to a python int: 
```python
clus = faiss.Clustering(10, int(ncent))
```

## How can diversity be enforced on search results?

Sometimes the results returned by queries on some index may be disappointing: if there are 100 instances of the same vector in the dataset, and a query happens to hit one of the instances, then the 99 other instances will fill the result list. From Faiss' point of view, this is the correct thing to do, because these are indeed the nearest neighbors of the query, but it is not satisfying from an application point of view.

Possible solutions:

- do not to add multiple instances of the same object to an index, 

- query more results than you need, and post-process the result list to remove duplicates and near duplicates.

## Is it possible to dynamically exclude vectors based on some criterion?

There is limited support for filtering vectors at search time, see [Searching in a subset of elements](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query#searching-in-a-subset-of-elements). 
Filtering must be based on the vector ids. 

Note that Faiss mainly relies on scanning strings of codes and computing distances. During the scan, it checks if the ID of a vector should be included into the result before computing the distance. This slows down the processing, so it is always more efficient to search on an index that contains just the relevant elements. 

For maximum speed, there are two workarounds that may be useful: 

- if only a small number of vectors should be ignore, query an index larger k than needed and filter out the irrelevant results post-hoc

- if the criterion is a discrete attribute with few distinct values, build one index per value of the attribute. 

## What does it mean when a search returns -1 ids?

To perform searches in an `IndexIVF` or `IndexHNSW`, the algorithm scans a subset of the elements. 
If there are not enough elements to fill the result list, the missing results are set to -1. 
You can increase the number of elements that are visited in an `IndexIVF` (resp. `IndexHNSW`) with `ParameterSpace().set_index_parameter(index, 'nprobe', 100)` (resp. `ParameterSpace().set_index_parameter(index, 'efSearch', 100)`). 
The parameter `nprobe`/`efSearch` adjusts the speed/accuracy tradeoff of the search.

However, note that if `nprobe` becomes `nlist` (the number of inverted lists), this is an exhaustive, exact (if using IVFFlat), brute-force search again and a Flat index will be faster in this case. `nprobe` should be substantially less than `nlist` in order to achieve speedup.

## Is PQ encoding with more or less than 8 bits per components supported?

Currently, the support is: 
- the `ProductQuantizer` object supports any code between `nbits=` 1 and 16. The code size of the M PQ indices is rounded up to a whole number of bytes, ie. PQ3x4 uses 2 bytes;
- the `IndexPQ` supports the same;
- the `IndexIVFPQ` supports the same (Faiss < 1.6.2 supports only PQ with 8 bits);
- the `MultiIndexQuantizer` supports up to 16 bits per code;

## How can I set nprobe on the sub-indexes of an `IndexShards` or `IndexReplicas`?

The `nprobe` field cannot be accessed directly, so you can either loop manually over the sub-indexes or use a `ParameterSpace` object. 

In C++: 

```
IndexShards index_shards = .....;
ParameterSpace().set_index_parameter(index_shards, "nprobe", 123);
```

In Python: 

```
index_shards = faiss.IndexShards(...)
ParameterSpace().set_index_parameter(index_shards, "nprobe", 123);
```

On GPU, `IndexShards` or `IndexReplicas` objects are built automatically by the `index_cpu_to_gpu*` functions. 
Instead of `ParameterSpace`, use `GpuParameterSpace`.

**WARNING**: setting `index_shards.nprobe = 123` in Python does not generate an error, but the nprobe of the index_shards will not be set. This is because Python can add fields to objects dynamically.

## How can I set nprobe on an "opaque" index?

Sometimes the `IndexIVF` is opaque (ie. seen by Python as an `Index` or as an Index* in C++). 
This is the case in particular when directly accessing `main_index.index` where `main_index` is an `IndexPreTransform`.

To set the nprobe there are two possibilities. 

- In C++: 
```
auto cpu_index = faiss::read_index(faissindex_file);
auto index_ivf = faiss::ivflib::extract_index_ivf(cpu_index);
index_ivf->nprobe = 123;
```
or 
```
auto cpu_index = faiss::read_index(faissindex_file);
ParameterSpace().set_index_parameter(cpu_index, "nprobe", 123);
```

- In Python:
```
cpu_index = faiss.read_index(faissindex_file)
index_ivf = faiss.extract_index_ivf(cpu_index)
index_ivf.nprobe = 123;
```
or 
```
cpu_index = faiss.read_index(faissindex_file)
ParameterSpace().set_index_parameter(cpu_index, "nprobe", 123)
```
If you use GPU indices, replace `ParameterSpace` with `GpuParameterSpace`. 

**WARNING**: setting `cpu_index.index.nprobe = 123` does not generate an error, but the nprobe of the index_ivf will not be set.
This is because `index.index` is seen by Python as a generic `Index` to which it adds field `nprobe` dynamically. 
This happens for any SWIG-wrapped object.

## Why does the RAM usage not go down when I delete an index?

The memory usage is usually measured by the Resident Set Size (RSS) of the process, which is returned by the Faiss function `get_mem_usage_kb` or top. 
For the RSS to decrease after an index is deleted, what must happen: 

1. (Python only) the refcount of the index must drop to 0. When that happens, the Python object is deleted, which almost always triggers a C++ delete. Make sure that there are no references to the index somewhere in the code, eg. `a=IndexFlatL2(10); b=a; del a` does *not* delete the object. 

2. The C++ delete calls the object's destructor, which deallocates the storage and returns the memory to the heap managed by the process via `free`. 

3. The memory manager of `libc` returns the memory to the system (via `sbrk` or `mmap`). This is not always possible due to fragmentation and in any case there are implementation and parameter choices involved on whether this will happen, see eg. http://man7.org/linux/man-pages/man3/mallopt.3.html for the glibc. 

Unlike C++, Python deallocation is not predictable. Depending on the Python version, it can be helpful to call `import gc; gc.collect()` to force a garbage collection cycle. This is especially useful for GPU indexes that eat up GPU memory even when the object is deleted.

## Why are the number of openmp threads not being set correctly when calling `faiss.omp_set_num_threads(n)`? 
The number of openmp threads is thread-local. This means when you call `faiss.omp_set_num_threads(n)` in thread A but the omp code is spawned from thread B, thread B is not going to be aware of the overriden openmp thread number. In order to circumvent this problem, either call `faiss.omp_set_num_threads(n)` from thread B or use the environment variable `OMP_NUM_THREADS` which is shared across threads to set the number of openmp threads.

