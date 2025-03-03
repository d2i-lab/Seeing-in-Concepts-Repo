There are a few additional operations that are not supported for all types of indexes.

# Database operations on indexes.

These are operations that rely on id-based access on the dataset. 

## Direct Map for an `IndexIVF`

The vector ids for an `IndexIVF` (and `IndexBinaryIVF`) are stored in the inverted lists. 
Therefore there is no way to map back from an id to the entry in the index. 
To support removal or updates on `IndexIVF`, the `DirectMap` field of the `IndexIVF` object stores a mapping from id to the location where it is stored in the index. 
It can have 3 values: 

- `DirectMap.NoMap`: no mapping is stored, reconstruction is not possible (default). 

- `DirectMap.Array`: the direct map is an array. The indices are assumed to be sequential, which rules out `add_with_ids`

- `DirectMap.Hashtable`: the direct map is a hashtable. Indices can be arbitrary and `add_with_ids` works (provided indices are distinct). 

Set or change the DirectMap type with `index.set_direct_map_type(DirectMap.Array)`.
Example usage here: [TestReconsHash](https://github.com/facebookresearch/faiss/blob/a17a631dc326b3b394f4e9fb63d0a7af475534dc/tests/test_index.py#L585).


## Reconstructing one vector or a subset

One vector can be reconstructed with the `reconstruct` method. 
To call it on several vectors (mainly useful in python to avoid the loop), use `reconstruct_batch`. 

## Reconstructing all vectors of a range: `reconstruct_n`

This function reconstructs all vectors from a range of contiguous ids. 
ids that are not present in the dataset are untouched on output.


## Reconstruct some vectors

Example usage: [test_index_composite.py](https://github.com/facebookresearch/faiss/blob/master/tests/test_index_composite.py)

This is supported for all types of indexes. Make sure to enable the direct map for the `IndexIVF` index types.

## Removing elements from an index

The method `remove_ids` removes a subset of vectors from an index. It takes an `IDSelector` object that is called for every element in the index to decide whether it should be removed. `IDSelectorBatch` will do this for a list of indices. The Python interface constructs this from numpy arrays if necessary.

NB that since it does a pass over the whole database, this is efficient only when a significant number of vectors needs to be removed (see exception below).

Example: [test_index_composite.py](https://github.com/facebookresearch/faiss/blob/master/tests/test_index_composite.py#L25)

Supported by `IndexFlat`, `IndexIVFFlat`, `IDMap`. 

Note that there is a semantic difference when removing ids from sequential indexes vs. when removing them from an `IndexIVF`: 

- for sequential indexes (`IndexFlat`, `IndexPQ`, `IndexLSH`), the removal operation shifts the ids of vectors above the removed vector id.

- the `IndexIVF` and `IndexIDMap2` store the ids of vectors explicitly, so the ids of other vectors are not changed. 
There are two special cases for `IndexIVF`: 
  - `DirectMap` type `Array` does not support removal because it means that all the indices would be shifted, which does not seem very useful.
   - with a direct map type `Hashtable` and a selector `IDSelectorArray` elements can be removed **without** scanning the whole index. 

# Range search

The method `range_search` returns all vectors within a radius around the query point (as opposed to the k nearest ones). Since the result lists for each query are of different sizes, it must be handled specially: 

- in C++ it returns the results in a pre-allocated [`RangeSearchResult`](https://github.com/facebookresearch/faiss/blob/master/impl/AuxIndexStructures.h#L35) structure 

- in Python, the results are returned as a triplet of 1D arrays `lims, D, I`, where result for query i is in `I[lims[i]:lims[i+1]]` (indices of neighbors), `D[lims[i]:lims[i+1]]` (distances).

Supported by (CPU only): `IndexFlat`, `IndexIVFFlat`, `IndexScalarQuantizer`, `IndexIVFScalarQuantizer`.

# Splitting and merging indexes

The methods: 

- `merge_from` copies another index to this and deallocates it on-the-fly. You can use `ivflib::merge_into` for `IndexIVF`s wrapped in a pre-transform. Please make sure you understand how vector ids are managed (sequential or user-assigned), see [This issue](https://github.com/facebookresearch/faiss/issues/3651#issuecomment-2265670699) for a discussion about the difference between the two. 

- `copy_subset_to` copies a subset of this codes to another index. Example usage: [to build indexes on a GPU and move them to CPU afterwards](https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_1bn.py#L541)

The functions are implemented only for `IndexIVF` subclasses because they are mainly interesting for large indexes. 
