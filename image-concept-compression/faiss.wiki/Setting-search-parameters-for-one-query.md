Faiss indexes have their search-time parameters as object fields. 

However, it can be useful to set these parameters separately per query. 
For example, for an `IndexIVF`, one query vector may be run with `nprobe=10` and another with `nprobe=20`. 
This is problematic when the searches are called from different threads. 

To support this use case, a `SearchParameter` object can be provided as the last argument of the `search()` function. 

## `SearchParameter` instances

Index types that accept search parameters have a corresponding `SearchParameter` child object. 
For example, `IndexIVFPQ` has a `SearchParameterIVFPQ` object. 

The search will look like: 
```python
D, I = index.search(xq, 10, params=faiss.SearchParametersIVFPQ(nprobe=10))
```
Note that the `params=` is mandatory, not to confuse the search parameters with possible `I` and `D` output buffers that can also be provided. 
In C++: 
```c++
idx_t *I = ...
float *D = ...
faiss::SearchParametersIVFPQ params; 
params.nprobe = 10; 
index.search(nq, xq, 10, D, I, &params);
```

SearchParameters can be nested, for example
```python
params=faiss.SearchParametersIVFPQ(
    nprobe=10, 
    quantizer_params=faiss.SearchParameterHNSW(efSearch=123)
)
```
is valid for an `IVF1234_HNSW,...` index. 

These functionalities are tested in [test_search_params.py](https://github.com/facebookresearch/faiss/blob/main/tests/test_search_params.py).

## Searching in a subset of elements

It is possible to select a subset of vectors, based on their ids, to search from. 
Similar to the [`remove_ids` method](https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#removing-elements-from-an-index), the subset is defined by an `IDSelector` object. 
The `IDSelector::is_member(i)` returns `true` if vector with id `i` should be included in the search. 

The `IDSelector` is provided via the `sel` field of `SearchParameters` (if it is non-null). 

For customization, in C++, it is possible to define an `IDSelector` subclass at will. 
In Python this is also possible (see below) but inefficient. 

By default, vectors are filtered using the `is_member` method but some combinations of index types and selector types are directly optimized specifically.
Below, we define the predefined `IDSelector` classes. 

### `IDSelectorRange`

This selects ids that are in range `[imin, imax)`. 

Specific optimizations: 

- for `IndexFlat`, ignores the data matrix outside of the `[imin, imax)` range 

- for `IndexIVF` with sorted ids, only the subsection of the inverted lists that contains the range is handled. 
The ids are sorted either if vectors are added with `add` or if `add_with_ids` was called with increasing ids (does not reqyure to be strictly). 
This can be checked with `index_ivf.check_sorted_ids()`.

- TODO: support for all flat indexes.


### `IDSelectorArray`, `IDSelectorBatch` and `IDSelectorBitmap`

These three `IDSelectors` encapsulate a set of elements to process. 
They differ by their properties, see below (n = nb of ids in index, k = nb of selected ids)

| class     | storage  | lookup cost    |
|-----------|----------|------------|
| IDSelectorArray | O(k) | O(k)  | 
| IDSelectorBatch | O(k) | O(1)  | 
| IDSelectorBitmap | O(n) | O(1)  |


For `IDSelectorArray`, the ids are just provided and stored as an array. 
Therefore, merely calling `is_member` can be slow. 
However, for some indexes (currently `IndexFlat`), it simply picks up the database elements to compare with, which is fast.

`IDselectorBatch` combines a hashtable (`unordered_set`) and a Bloom filter. 
Thus the lookup is in constant time, but there is some memory overhead.

`IDSelectorBitmap` takes an array with 1 bit per vector id. 
It is particularly interesting when a significant subset of vectors needs to be handled and ids are sequential. 
Then it is more compact and faster than `IDselectorBatch`.

`IDSelectorBatch` and `IDSelectorArray` are constructed from an array of ids to store. 
`IDSelectorBitmap` takes the binary mask as an `uint8` array as argument. 


### `IDSelectorNot`

This reverts another IDSelector. For example: 
```python
sel = faiss.IDSelectorNot(faiss.IDSelectorBatch(list_of_ids))
```
ignores the ids from the list. 

### `PyCallbackIDSelector`

This wraps a Python function and exposes it at the `is_member`. 
The Python function must take an `int` and return a `bool`.  

```python
value = 123

def my_condition(i): 
    return i % value == 0 

sel = faiss.PyCallbackIDSelector(my_condition) 
```

This is inefficient because the Python function is called for every database entry (after capturing the GIL), but useful for debugging. 

### Time Complexity

Ids are not pre-filtered but rather checked as we traverse codes and ids in the ivf scanner except IDSelectorRange which allows limiting the search from the beginning if ids are sorted in scanner. For example, Time complexity for IDSelectorRange on IndexIVF with sorted ids is O(k) otherwise O(n) if ids are random
For time complexity, please see below (n = nb of ids in index, k = nb of selected ids)

| class     | Time Complexity  | Specific optimizations | 
|-----------|----------|------| 
| IDSelectorRange | O(n) | O(k) for `IndexFlat`, O(k) for `IndexIVF` with `assume_sorted=true` |
| IDSelectorArray | O(n * k) | O(k) for `IndexFlat` |
| IDSelectorBatch | O(n * log(n)), worst case |
| IDSelectorBitmap | O(n) | | 
