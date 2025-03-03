Pre- and post-processing is used to: remap vector ids, apply transformations to the data, and re-rank search results with a better index.

## Faiss ID mapping 

By default Faiss assigns a sequential id to vectors added to the indexes. This page explains how to change this to arbitrary ids.

Some `Index` classes implement a `add_with_ids` method, where 64-bit vector ids can be provided in addition to the the vectors. 
At search time, the class will return the stored ids rather than the sequential vector ids. 

<!-- WARNING: in Lua, the search function adds 1 to the search results by default, to account for Lua 1-based indexing. Calling the search function with `search (queries, k, false)` cancels this behaviour and return the raw ids.  -->

### The `IndexIDMap` 

This index encapsulates another index and translates ids when adding and searching. It maintains a table with the mapping.

For example, 

```python
index = faiss.IndexFlatL2(xb.shape[1]) 
ids = np.arange(xb.shape[0])
index.add_with_ids(xb, ids)  # this will crash, because IndexFlatL2 does not support add_with_ids
index2 = faiss.IndexIDMap(index)
index2.add_with_ids(xb, ids) # works, the vectors are stored in the underlying index
```

### IDs in the `IndexIVF`

The `IndexIVF` sub-classes always store vector IDs. Therefore, the `IndexIDMap`'s additional table is a waste of space. The `IndexIVF` offers `add_with_ids` natively. 

## Pre-transforming the data 

It is often useful to transform data prior to indexing. Transformation classes inherit `VectorTransform`. A `VectorTransform` applies a transformation to input vectors (of dimension `d_in`) and outputs vectors of size `d_out`. 

| Transformation                     | Class name      | Comments            
|------------------------------------|-----------------|---------------------
| random rotation                    | `RandomRotationMatrix`  | useful to re-balance components of a vector before indexing in an `IndexPQ` or `IndexLSH` 
| remapping of dimensions            | `RemapDimensionsTransform` | to reduce or increase the size of a vector because the index has a preferred dimension, or to apply a random permutation on dimensions. 
| PCA                                | `PCAMatrix`       | for dimensionality reduction |
| OPQ rotation                       | `OPQMatrix`       | OPQ applies a rotation to the input vectors to make them more amenable to PQ coding. See [Optimized product quantization, Ge et al., CVPR'13](http://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Ge_Optimized_Product_Quantization_2013_CVPR_paper.html) for more details.


Transformations can be trained from a set of vectors if it makes sense, using the method `train`. They can be applied to a set of vectors with `apply`. 

An index can be wrapped in a `IndexPreTransform` index so that the mapping occurs transparently, and training is integrated with the Index training. 


## Example: apply a PCA to reduce the number of dimensions 

### Standalone version 

See the next section

### With an IndexPreTransform 

For example, if the input vectors are 2048D, and have to be reduced to 16 bytes, it makes sense to reduce them with a PCA before.

```python
  # the IndexIVFPQ will be in 256D not 2048
  coarse_quantizer = faiss.IndexFlatL2 (256)
  sub_index = faiss.IndexIVFPQ (coarse_quantizer, 256, ncoarse, 16, 8)
  # PCA 2048->256
  # also does a random rotation after the reduction (the 4th argument)
  pca_matrix = faiss.PCAMatrix (2048, 256, 0, True) 

  #- the wrapping index
  index = faiss.IndexPreTransform (pca_matrix, sub_index)

  # will also train the PCA
  index.train(...)
  # PCA will be applied prior to addition
  index.add(...)
```

### Example: increase the number of dimensions

It is sometimes useful to increase the number of dimensions d by inserting zeros in the vectors. This can be useful for:

- make `d` a multiple of 4, this is what the distance computations are optimized for

- make `d` a multiple of `M`, where `M` is the size of a PQ.

This can be done with

```python
  # input is in dimension d, but we want a multiple of M
  d2 = int((d + M - 1) / M) * M
  remapper = faiss.RemapDimensionsTransform (d, d2, true)
  # the index in d2 dimensions  
  index_pq = faiss.IndexPQ(d2, M, 8)  
  
  # the index that will be used for add and search 
  index = faiss.IndexPreTransform (remapper, index_pq)
```

## `IndexRefineFlat`: re-ranking search results 

When querying a vector, it may be useful to re-rank the search results with real distance computations. The following example searches an index with an `IndexPQ`, then reranks the first results by computing real distances: 

```python
  q = faiss.IndexPQ (d, M, nbits_per_index)
  rq = faiss.IndexRefineFlat (q)
  rq.train (xt)
  rq.add (xb)
  rq.k_factor = 4
  D, I = rq:search (xq, 10)
```

The search function will fetch 4 * 10 nearest neighbors from the IndexPQ, then compute the real distance for each of the results, and keep the 10 best ones. Note that the `IndexRefineFlat` has to store the full vectors, so it is not memory-efficient. 

## IndexShards: combining results from several indexes 

When the dataset is spread over several indexes, a query can be dispatched over them and the results can be combined with an `IndexShards`. This is also useful if the index is spread over several GPUs and the queries can be done in parallel, see `index_cpu_to_gpus` with `shards` set to true in `GpuClonerOptions`. 

