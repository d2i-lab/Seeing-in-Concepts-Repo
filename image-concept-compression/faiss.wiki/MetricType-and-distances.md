# Notes on `MetricType` and distances

There are two primary methods supported by Faiss indices, L2 and inner product. Others are supported by `IndexFlat`.

For the full list of metrics, [see here](https://github.com/facebookresearch/faiss/blob/master/faiss/MetricType.h#L44).

## METRIC_L2

Faiss reports squared Euclidean (L2) distance, avoiding the square root. This is still monotonic as the Euclidean distance, but if exact distances are needed, an additional square root of the result is needed.

This metric is invariant to rotations of the data (orthonormal matrix transformations). 

## METRIC_INNER_PRODUCT

This is typically used for maximum inner product search in recommendation systems. 
The norm of the query vectors does not affect the ranking of results (of course the norm of the database vectors *does* matter). 
This is not by itself cosine similarity, unless the vectors are normalized (lie on the surface of a unit hypersphere; see cosine similarity below).

## How can I index vectors for cosine similarity?

The cosine similarity between vectors $x$ and $y$ is defined by:

$$
\cos(x, y) = \frac{\langle x, y \rangle}{\|x\| \times \|y\|}
$$

It is a similarity, not a distance, so one would typically search vectors with a *larger* similarity.  

By normalizing query and database vectors beforehand, the problem can be mapped back to a maximum inner product search.
To do this: 
- build an index with `METRIC_INNER_PRODUCT`
- normalize the vectors prior to adding them to the index (with `faiss.normalize_L2` in Python)
- normalize the vectors prior to searching them 

Note that this is equivalent to using an index with `METRIC_L2`, except that the distances are related by $\| x - y \|^2 = 2 - 2 \times \langle x, y \rangle$ for normalized vectors.

## Additional metrics

Additional metrics are supported by `IndexFlat`, `IndexHNSW` and `GpuIndexFlat`.

[`METRIC_L1`](https://en.wikipedia.org/wiki/Taxicab_geometry), [`METRIC_Linf`](https://en.wikipedia.org/wiki/Chebyshev_distance) and [`METRIC_Lp`](https://en.wikipedia.org/wiki/Lp_space) metrics are supported. `METRIC_Lp` includes use of `Index::metric_arg` (C++) / `index.metric_arg` (Python) to set the power.

[`METRIC_Canberra`](https://en.wikipedia.org/wiki/Canberra_distance), [`METRIC_BrayCurtis`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.braycurtis.html) and [`METRIC_JensenShannon`](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) are available as well. For Mahalanobis see below.

## How can I index vectors for Mahalanobis distance?

The [Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) is equivalent to L2 distance in a transformed space. 
To transform to that space: 
- compute the covariance matrix of the data 
- multiply all vectors (query and database) by the inverse of the Cholesky decomposition of the covariance matrix. 
- index in a METRIC_L2 index.

Example here: [mahalnobis_to_L2.ipynb](https://gist.github.com/mdouze/6cc12fa967e5d9911580ef633e559476)

How to whiten data with Faiss and compute Mahalnobis distance: [demo_whitening.ipynb](https://gist.github.com/mdouze/33fc39927c343c4ca003f1d8f5a412ef)

## How can I do max Inner Product search on indexes that support only L2?

Vectors can be transformed by adding one dimension so that max IP search becomes equivalent to L2 search. See [Speeding up the xbox recommender system using a euclidean transformation for inner-product spaces, Bachrach et al, ACM conf on recommender systems, 2014](http://ulrichpaquet.com/Papers/SpeedUp.pdf) section 3 that transforms inner product computations into L2 distance computations. 
See an implementation here [demo_IP_to_L2.ipynb](https://gist.github.com/mdouze/e4bdb404dbd976c83fe447e529e5c9dc).

Note however that while mathematically equivalent, this may not interact well with quantization (and possibly other index structures, see [Non-metric similarity graphs for maximum inner product search, Morozov & Babenko, Neurips'18](https://proceedings.neurips.cc/paper/2018/hash/229754d7799160502a143a72f6789927-Abstract.html))

## How can I do L2 searches for indexes that support only max inner product? 

The reverse transformation also uses an additional dimension to the vector, see [Asymmetric Mapping Quantization
for Nearest Neighbor Search, Hong et al, PAMI'20](https://cse.buffalo.edu/~jsyuan/papers/2020/Asymmetric_Mapping_Quantization_for_Nearest_Neighbor_Search.pdf). 

## How can I find the farthest away vector instead of the nearest one? 

For cosine similarity and inner product, just query the opposite vector. 

For L2, there is a trick that involves one additional dimension: 
[demo_farthest_L2.ipynb](https://gist.github.com/mdouze/c7653aaa8c3549b28bad75bd67543d34#file-demo_farthest_l2-ipynb)

