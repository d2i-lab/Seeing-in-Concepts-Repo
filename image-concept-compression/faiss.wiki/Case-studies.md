In this page, we reference example use cases for Faiss, with some explanations. 
The examples will most often be in the form of Python notebooks, but as usual translation to C++ should be smooth. 

## Implementing an evolving IVF dataset 

This script demonstrates how to add/remove elements from an IVF dataset in a rolling fashion. The key is to use a Hashtable as DirectMap type and remove with IDSelectorArray. Removal cost is then proportional to the number of elements to remove instead of number of elements in the dataset.

[demo_rolling_dataset.ipynb](https://gist.github.com/mdouze/430a67fbe0937482a1fd537e14c51af0#file-demo_rolling_dataset-ipynb)


## Fast indexing of 2M vectors for max inner product search

This script demonstates how to speed up a recommendation system.
Conceptually, the queries vectors are users and the database vectors are items to recommend. The metric to "compare" them is maximum inner product, ie. which item is the most relevant for each user.
There is a real-time constraint for this use case (should be returned in < 5 ms) and the accuracy should be as high as possible.

[recommendation_2M.ipynb](https://gist.github.com/mdouze/eedf3d9afe78d09e5ee42e1931052b05#file-recommendation_2m-ipynb)

## Limited size clustering 

This script demonstrates how to do a k-means variant where in addition the clusters are constrained to contain no more than a maximum number of points.

[limited_size_clustering.ipynb](https://gist.github.com/mdouze/9126b72eb9a4b8bd6f624e20649edf73#file-limited_size_clustering-ipynb)

## Asymmetric binary search

This script demonstrates an asymmetric search use case: the query vectors are in full precision and the database vectors are compressed as binary vectors. This implementation is slow, it is mainly intended to show how much accuracy can be regained with asymmetric search.

[demo_asymmetric_binary.ipynb](https://gist.github.com/mdouze/b2e6c6144d4e06fca8287f5257f15fed#file-demo_asymmetric_binary-ipynb)

## Manual training of IVFPQ 

This script demonstrates how to manually train an IVFPQ index enclosed in a OPQ pre-processor. This can be useful, for example, if there are pre-trained centroids handy for the data distribution.

This is also implemented in the function [train_ivf_index_with_2level](https://github.com/facebookresearch/faiss/blob/main/contrib/clustering.py#L86). It should be easy to expand to other types of composite indexes.

[manual_IVFPQ_training.ipynb](https://gist.github.com/mdouze/8f43d37037d0ca19327539c0f8227f8e#file-manual_ivfpq_training-ipynb)

## Mixed sparse-dense clustering 

There is a sparse clustering implementation in `faiss.contrib.clustering`.
This script demonstrates how to cluster vectors that are composed of a dense
part of dimension d1 and a sparse part of dimension d2 where d2 >> d1. 
The centroids are represented as full dense vectors. 

The implementation relies on the `clustering.DatasetAssign` object, that abstracts
away the representation of the vectors to cluster. The `clustering` module contains 
a pure Python implementation of `kmeans` that can consume this `DatasetAssign`. 

[sparse_dense_clustering.ipynb](https://gist.github.com/mdouze/f87ac3b5c66a6a0c3cfb8fbb59ff52e8#file-sparse_dense_clustering-ipynb)

## Separate coarse quantization

This script demonstrates how to use a high-dimensional coarse quantizer with a low-dimensional fine quantizer. 
This is not possible out-of-the-box because the IVFPQ implementation assumes the IVF quantizer and the PQ run in the same dimension. 
To combine the quantizers in different dimensionalities, the approach is to use `search_preassigned` and `add_preassigned` to perform the coarse quantization and add / search separately.  

In this example, the OPQ pre-transformation (an orthonormal transformation of the data) reduces the dimension of the input from 96 to 32 dimensions, so the coarse quantizer may not be as selective as it could.
By doing the coarse quantization and the search separately, the accuracy improves for some (but not all) settings of `nprobe`. 

[demo_independent_ivf_dimension.ipynb](https://gist.github.com/mdouze/8c5ab227c0f7d9d7c15cf92a391dcbe5#file-demo_independent_ivf_dimension-ipynb)

## Manual implementation of IVFPQ search 

This script demonstrates how to peform IVFPQ search in Python manually. 

[demo_ivfpq_distance_tables.ipynb](https://gist.github.com/mdouze/08b3aaec6bf4a82994d89cb955ac1c57#file-demo_ivfpq_distance_tables-ipynb)

## Sharded GPU dataset with simultaneous queries (pytorch)

This example script shows how to handle a database sharded over n GPUs. 
Each GPU issues a set of queries simultaneously. The queries are performed 
over the sharded dataset and the results are sent back to the issuing GPU. 
This is typical for pytorch training jobs, that need to do searches at each 
iteration over a dataset that is scattered around the training GPUs. 

[bench_faiss_n2n.py](https://gist.github.com/mdouze/93854e55e210a03c9ca3475b09d7c3e7)

