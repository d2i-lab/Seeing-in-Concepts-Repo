It is not always obvious how the C++ and Python layers interact. 
Therefore, we give some handy code in Python notebooks that can be copy/pasted to perform some useful operations.

They rely mostly on `vector_to_array` and a few other Python/C++ tricks described [here](https://github.com/facebookresearch/faiss/wiki/Faiss-code-structure#c-code-wrapping)

The [`faiss.contrib.inspect_tools`](https://github.com/facebookresearch/faiss/blob/master/contrib/inspect_tools.py) module has a few useful functions to inspect the Faiss objects. 
In particular `inspect_tools.print_object_fields` lists all the fields of an object and their values.

## How can I get the PCA matrix in numpy from a PCA object?

Use the function `faiss.contrib.inspect_tools. get_LinearTransform_matrix `, or see this code: 
[get_matrix_from_PCA.ipynb](https://gist.github.com/mdouze/ca65bce66f77cd2ef4df8769e19443a9).
This applies to any `LinearTransform` object. 

## How can I get / change the centroids from a ProductQuantizer or ResidualQuantizer object?

For PQ: see [access_PQ_centroids.ipynb](https://gist.github.com/mdouze/72221ab1937baa219545aff9ba7221c0). 

For RQ: see [demo_replace_RQ_codebooks.ipynb](https://gist.github.com/mdouze/575641f4d14116baea6083fd0587f7a2#file-demo_replace_rq_codebooks-ipynb)

## How can I get the content of inverted lists?

Use the function `faiss.contrib.inspect_tools.get_invlist`, or see this code: 
[get_invlists.ipynb](https://gist.github.com/mdouze/b0a65aba70a64b00d6425b0e268a4c80)

## How can I lookup the inverted list corresponding to a stored vector? 

This does not require C++ magic. See [#3555](https://github.com/facebookresearch/faiss/discussions/3555)


## How can I get the link structure of a HNSW index?

See this code snippet: [demo_hnsw_struct.ipynb](https://gist.github.com/mdouze/7d5271e49a3d4b8c9c8d1eac8f4b9748)
[alternative rendering](https://gist.github.com/mdouze/42c10ff901970d94ddb1cc182e392f2a).

## How can I get the knn graph for an `IndexNNDescent`?

See 
[demo_access_nndescent.ipynb](https://gist.github.com/mdouze/105dd68d8921134e9270e832f4bdbe66#file-demo_access_nndescent-ipynb)

## How can I merge normal ArrayInvertedLists?

See [demo_merge_array_invertedlists.ipynb](https://gist.github.com/mdouze/7331e6fc1da2334f30706b9b9962068b)

## Faiss/pytorch interop: how can I use a PQ codec without leaving the GPU?

See [PQ_codec_pytorch.ipynb](https://gist.github.com/mdouze/94bd7a56d912a06ac4719c50fa5b01ac).

## How to explore the contents of an opaque index?

We have an index file but don't know what's in it. 
When accessing the `Index` fields of a wrapper index, they show up as a plain `Index` object. 
The `downcast_index` converts this plain index to the "leaf" class the index belongs to.
This snippet is a demo on how to use `downcast_index` to extract all info from it: 
[demo_explore_indedex.ipynb](https://gist.github.com/mdouze/95b4bb74dbb4c7e2387efc14486e23ba)

## How can I get all the ids from an IDMap or an IDMap2?

`IDMap2` inherits `IDMap`, so [this code](https://gist.github.com/mdouze/fa6a2951fecb6d965a2aa66d20474a93) works for both.

## how can I convert an IDMap2 to IDMap?

This code works for both directions: 
[convert_idmap2_idmap.ipynb](https://gist.github.com/mdouze/773b2e1b42ac50f700407f3a727921e5)

## How to train a CPU index with a GPU just for k-means?

See [train_ivf_with_gpu.ipynb](https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6)


<!-- 

## How to use a GPU for PQ and OPQ training?

See [train_opq_with_gpu.ipynb](https://gist.github.com/mdouze/11869525966715e84967e510b1e33229).
-->

## How to use the GPU at add time?

See [assign_on_gpu.ipynb](https://gist.github.com/mdouze/334ad6a979ac3637f6d95e9091356d3e).

## How can I force the k-means initialization?

plus: how to do this for IVF training

See [initial_centroids_demo.ipynb](https://gist.github.com/mdouze/9eb96d941c94ef59482a069e5862a650)

## How to transfer a trained OPQ and/or IVF centroids to another index? 

See https://github.com/facebookresearch/faiss/issues/2455

## How can I replace the inverted list content: 

See 
[demo_replace_invlists.ipynb](https://gist.github.com/mdouze/aef2078afdb12c027ed93672d9801399#file-demo_replace_invlists-ipynb)

## How can I get access to non-8 bit quantization code entries in PQ / IVFPQ / AQ ?

You need a `BitStringReader`, see [#2285](https://github.com/facebookresearch/faiss/issues/2285#issuecomment-1087228404)


## Simulating an IndexPQ on GPU with a 1-centroid IVFPQ

IndexPQ is not supported on GPU, but it is relatively easy to simulate it with an IVFPQ. 

[demo_1_centroid_PQ.ipynb](https://gist.github.com/mdouze/1627af05a2e064a6280f9ba1fb04934e#file-demo_1_centroid_pq-ipynb)

## Wrapping small C++ objects for use from Python

Sometimes it is useful to implement a small callback needed by Faiss in C++. 
However, it may be too specific or depend to external code, so it does not make sense to include in Faiss (and Faiss is hard to compile ;-) )

In that case, you can make a [SWIG](swig.org/Doc4.0/Sections.html#Sections) wrapper for a snippet of C++. 

Here is an example for an [`IDSelector`](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query#searching-in-a-subset-of-elements) object that has an is_member callback: [bow_id_selector.swig](https://gist.github.com/mdouze/fd2414bf2ed1e436ce80b5d30e48996b#file-bow_id_selector-swig)

To compile the code with Faiss installed via conda and SWIG 4.x on Linux: 

```bash 

# generate wrapper code 
swig -c++ -python -I$CONDA_PREFIX/include  bow_id_selector.swig 

# compile generated wrapper code: 
g++ -shared -O3 -g -fPIC bow_id_selector_wrap.cxx -o _bow_id_selector.so  \
  -I $( python -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())" )  \
  -I $CONDA_PREFIX/include $CONDA_PREFIX/lib/libfaiss_avx2.so

```
This produces `bow_id_selector.py` and `_bow_id_selector.so` that can be loaded in Python with 

```python 
import numpy as np
import faiss
import bow_id_selector

# very small sparse CSR matrix
n = 3
indptr = np.array([0, 2, 3, 6], dtype='int32')
indices = np.array([7, 8, 3, 1, 2, 3], dtype='int32')

# don't forget swig_ptr to convert from a numpy array to a C++ pointer
selector = bow_id_selector.IDSelectorBOW(n, faiss.swig_ptr(indptr), faiss.swig_ptr(indices))

selector.set_query_words(1, 2)
selector.is_member(0)   # returns False
selector.is_member(1)   # returns False
selector.is_member(2)   # returns True
selector.is_member(3)   # crashes! 

# And of course you can combine it with existing Faiss objects
params = faiss.SearchParameters(sel=selector)

```