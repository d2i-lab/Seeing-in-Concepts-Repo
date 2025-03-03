In the case of `IndexFlat`, the added data is just copied to the index without further encoding or organization. 
Therefore, it may be useful to short-circuit the indexing altogether. 
This makes it possible to: 

- avoid having two copies of the same data in memory

- do updates on the data between searches. 

Faiss provides low-level functions to do the brute-force search in this context. 

The functions take a matrix of database vectors and a matrix of query vectors and return the k-nearest neighbors and their distances. 

## Brute force search on CPU 

On CPU, the relevant function is `knn_L2sqr` or `knn_inner_product`, see [utils/distances.h](https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances.h#L200)

In Python, these functions are exposed as `contrib.exhaustive_search.knn`, see [tests/test_contrib.py](https://github.com/facebookresearch/faiss/blob/master/tests/test_contrib.py#L66).

## Brute force search on GPU 

On GPU, the relevant function is `bfKnn`. 
An additional advantage is that it takes both CPU and GPU resident data as input or output. 

Note that on GPU, the synchronization is needed because Faiss uses a non-default CUDA stream. 
The easiest workaround is to force it to use the default stream. This is done via  
```
res = faiss.StandardGpuResources()
res.setDefaultNullStreamAllDevices()
```
Also, by default the amount of memory allocated by the resource object is too large for simple brute force. You can set: 
```
res.setTempMemory(64 * 1024 * 1024)
```
ie. use only 64M (if that is not enough, try a few other values, according to some reports setting to 0 works just fine).

In Python there are two function wrappers that expose this functionality: 

- for numpy data, use `contrib.exhaustive_search.knn_gpu`, tested in [gpu/test_contrib.py](https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/test/test_contrib.py#L34)

- for pytorch data, use `contrib.pytorch_tensors.search_raw_array_pytorch`

An example usage in python, from pytorch GPU data is here: [test_pytorch_faiss.py](https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/test/test_pytorch_faiss.py)

## Brute force search _without Faiss_

This demonstrates how to do brute force search without using Faiss at all: 

- In numpy [demo_numpy_knn.ipynb](https://gist.github.com/mdouze/a8c914eb8c5c8306194ea1da48a577d2)

- in pytorch (GPU optional) [demo_pytorch_knn.ipynb](https://gist.github.com/mdouze/551ef6fa0722f2acf58fa2c6fce732d6#file-demo_pytorch_knn-ipynb).


## Combining the results from several searches

When the number of query vectors is limited, the best indexing method is to not index at all and use brute force search. 
This can be problematic when the dataset vectors do not fit in RAM or GPU RAM. 
In that case, it is efficient to search the `xq` vectors in slices `xb` of the database vectors. 
To maintain the top-k nearest neighbors seen so far, use a `ResultHeap` structure: 

https://github.com/facebookresearch/faiss/blob/master/benchs/bench_all_ivf/datasets.py#L75