This page contains a few sparse performance tips to make Faiss run faster, including the speed-vs-accuracy tradeoffs where the speed is more important.

# Speed vs. accuracy index parameters

Here is a summary of parameters to look at when configuring an index.

## Pick the index family

Start from picking the index [for similarity search](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) or [for vector codec](https://github.com/facebookresearch/faiss/wiki/Vector-codecs) purposes.

Additionally, please read the information about [additive quantizers](https://github.com/facebookresearch/faiss/wiki/Additive-quantizers).

<!-- 
Additionally, there is a family of ProductAdditiveQuantizer-based indices that is not fully documented and benchmarked yet :(, but we're working on it. 
Matthijs: leaving it out since we did not document it yet ;-) 
--> 

## Pick the index parameters

It is very common to instantiate an index via `faiss.index_factory()` call. But this way of instantiating sets the index parameters to safe values, while there are many speed-related parameters. Many of Faiss components may utilize:
* iterative algorithms used at encoding time; the more iterations performed, the better the recall rate is.
  * HNSW: `efConstruction`
  * ResidualQuantizer: `niter_codebook_refine`
  * LocalSearchQuantizer: `encode_ils_iters`, `icm_iters`, `train_iters`, `encode_ils_iters` (for vector codecs)
* heuristics that limit the number of possible candidates considered during the training or the search process, such as [beam search](https://en.wikipedia.org/wiki/Beam_search). The more candidates are evaluated, the better the recall rate is.
  * IndexBinaryIVF: `nprobe`, `max_codes`
  * IndexIVF: `nprobe`, `max_codes`
  * ResidualQuantizer: `max_beam_size`
  * HNSW: `efSearch`
* additional optional versions, such as LUT-based approximations; they allow to trade the speed for accuracy
  * IndexIVFPQ: `do_polysemous_training`, `polysemous_ht`, `use_precomputed_table`
  * IndexIVFPQFastScan: `use_precomputed_table`
  * IndexPQ: `do_polysemous_training`, `polysemous_ht`, `search_type`
  * ResidualQuantizer: `use_beam_LUT`, `approx_topk_mode`, `train_type`
  * LocalSearchQuantizer: `update_codebooks_with_double`
* computations that are optimized for a particular case, such as [big-batch IVF search](https://github.com/facebookresearch/faiss/pull/2567).
* various internal buffers for processing an input data in chunks; see below.
* random number generators; these ones are used to ensure that results can be reproduced.

**The default index parameters should work reasonably good in the most general cases.** But it is highly recommended to study the available parameters in order to find an optimal point in your particular speed-vs-accuracy-vs-RAM case.

**The recipe:** study the available parameters of a chosen index and its subindices.

## K-means clustering

K-means clustering is an often used facility inside Faiss. By default, k-means implementation in [faiss/Clustering.h](https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h) uses 25 iterations (`niter` parameter) and up to 256 samples from the input dataset per cluster needed (`max_points_per_centroid` parameter). The samples are chosen randomly. For example, the default `PQx12` training is ~4x slower than `PQx10` training and ~16x slower than `PQx8` training because of the proportionally higher number of used samples from the input dataset.

IVF-based indices use 10 iterations of k-means by default ([faiss/IndexIVF.h](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexIVF.h), [faiss/IndexBinaryIVF.h](https://github.com/facebookresearch/faiss/blob/main/faiss/IndexBinaryIVF.h), [faiss/gpu/GpuIndexIVF.cu](https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/GpuIndexIVF.h)).

**The recipe:** play with parameters of k-means clustering `niter` and `max_points_per_centroid`.

## Query batches

Faiss is optimized for working with batches of samples, rather than processing samples one by one.
Internally, Faiss parallelizes over the batch elements in a way that is more efficient than if parallelization was performed by the caller. 

**The recipe:** try to process your samples in batches, if appropriate.

## `Train_shared` training mode for PQ training

Consider using `Train_shared` training mode for PQ, which is used to train a single codebook for all subquantizers, instead of training a separate one for every subquantizer. This significantly reduces the training time for, say, 10-bit or 12-bit PQ, but at the cost of the accuracy. 

```c++
faiss::IndexPQ indexPQ;
indexPQ.pq.train_type = faiss::ProductQuantizer::Train_shared;

faiss::IndexIVFPQ indexIVFPQ;
indexIVFPQ.pq.train_type = faiss::ProductQuantizer::Train_shared;
```

# System configuration

These tips are about the environment of Faiss: threading, memory configuration, etc.

## Huge memory pages

Most modern CPU architectures and operating systems support [Huge memory pages](https://en.wikipedia.org/wiki/Page_(computer_memory)#Multiple_page_sizes).

It is a typical situation that Faiss uses the random access pattern to access the needed data, such as codebooks. As a result, TLB cache operates under a very heavy pressure. It was observed that large memory pages helps in certain cases. In particular, the use of 2M / 1G pages allowed to gain up to 20% speed in our certain vector codec experiments on x86-64 platform. Faiss does not contain a built-in support for marking certain allocated memory blocks as huge-pages-are-important-here ones. So, the simplest way to try out huge memory pages in your application is to use a memory allocator that support ones, such as [mimalloc](https://github.com/microsoft/mimalloc).

We did not conduct any experiments with huge memory pages on ARM platform yet.

**The recipe:** try a memory allocator that supports huge memory pages and try to run your application with ones being enabled. Please be aware that huge memory pages is a scarce resources and it needs to be managed carefully.

## NUMA nodes

Faiss does not have a NUMA-aware code. This means that Faiss does not coordinate memory allocations in order to minimize the traffic between the NUMA nodes. As an example, we had an observation where two runs of the same benchmark on a single 12C/24T NUMA node and on four NUMA nodes on the same machine would yield the _same_ running time!

**The recipe:** try playing with the number of cpu resources that you allocate for your application. 

## Intel MKL library

Faiss uses [Intel MKL library](https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-mkl-for-dpcpp/top.html), if possible. MKL uses OpenMP library under the hood. According to the MKL manual, it runs the best if the number of used threads equals to the number of physical cores ([for example](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/MKL-no-of-threads-vs-no-of-processor/td-p/1072982)), which is in agreement with our benchmarks. The default setting is that it uses the number of hyperthreads (twice the number of cores). 

**The recipe:** try setting the number of used OMP threads to the number of physical CPU cores, unless you're using a version of Faiss without any multithreading. So, on a 12C/24T machine run your application as `OMP_NUM_THREADS=12 my-application`.

## Sizes of various internal buffers

If a huge dataset is passed as an input, then it might be possible that the dataset will be split into chunks and processed chunk-by-chunk under the hood. It is possible to speedup the overall processing by increasing the size of a chunk, thus increasing the effectiveness of cache and linear algebra operations. On the other hand, it increases the amount of RAM for the allocated chunks and the time that is needed for such memory allocations. Here is the list of various internal buffer sizes that one may vary:

* `IndexBinaryFlat.query_batch_size`
* `LocalSearchQuantizer.chunk_size`
* `AdditiveQuantizer.max_mem_distances`
* `faiss.multi_index_quantizer_search_bs`
* `faiss.index2layer_sa_encode_bs`
* `faiss.product_quantizer_compute_codes_bs`
* `faiss.index_ivfpq_add_core_o_bs`
* `faiss.precomputed_table_max_bytes`
* `faiss.hamming_batch_size`
* `faiss.rowwise_minmax_sa_encode_bs`
* `faiss.rowwise_minmax_sa_decode_bs`
* `faiss.distance_compute_blas_threshold`
* `faiss.distance_compute_blas_query_bs`
* `faiss.distance_compute_blas_database_bs`
* `faiss.distance_compute_min_k_reservoir`

<!-- Unfortunately, some internal buffers are not currently exposed. --> 

## AVX-512 support 

Faiss uses custom kernels that involve the usage AVX2 or ARM NEON intrinsics. Currently, Faiss does not provide specialized AVX-512 kernels (except for [one](https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_fused/avx512.h), which must be enabled manually). 
We consider supporting AVX-512 more widely, but: 
* Faiss may use Intel MKL library (if available) that is capable of using AVX-512 under the hood (if MKL decides to). So, expensive GEMM-related operations are accelerated via AVX-512.
* AVX-512 computations lead to CPU downclocking, which may result in a lower performance (compared vs AVX2 version).
* AVX-512 kernels need developer time and dedication, and kernels may be tricky to create and benchmark.
* AVX-512 would increase the compilation time on our end, because it would introduce a new set of binaries.

As of today (Jan 2023):
* AVX-512 downclocking seems to be fixed in new generations of Intel CPUs. 
* AMD Zen4 seems to provide support for AVX-512 instructions.
* [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) instructions come to AVX-512, which can be used for certain cases, such as Additive Quantizers. But the corresponding support needs to be added and benchmarked first.

<!-- So, it is possible that Faiss will add some AVX-512 kernels in the future. --> 

Faiss may use Intel MKL library, which allows to enable/disable certain instruction sets via [the following](https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/support-functions/miscellaneous/mkl-enable-instructions.html), if it fits your case and/or if the frequency downclocking is not beneficial.

The only AVX-512 provided custom kernel so far is a [fused kernel](https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_fused/avx512.h) that combines the L2 distance computation and the closest nearest search neighbor search. This kernel needs to be enabled manually. It may provide some additional minor speedup for PQ training or k-means clustering in case of small dimensionality values (1-16).

# Transposed centroid table for Product Quantizer 

(Faiss 1.7.3+)

Allows to trade some very minor amount of RAM for a speedup of `pq.search()` call.
[The reference in wiki](https://github.com/facebookresearch/faiss/wiki/Implementation-notes#transposed-centroids-table-for-pq-faiss-173)

This can be used for CPU-based IndexPQ and IndexIVFPQ as well, for example:
```python
# train an IVFPQ CPU index
...

# create the tables
index.pq.sync_transposed_centroids()

# do the search
while app_is_running:
  D, I = index.search(xq, k)
```

# Vector codecs

Fast vector decoding facilities were [added](https://github.com/facebookresearch/faiss/wiki/Vector-codecs#experimental-high-performance-decoder-kernels) in the form of C++ templates for certain codecs.

To speed up training, do not use [Polysemous codes](https://github.com/facebookresearch/faiss/wiki/The-index-factory) for vector codec purposes. For example, use `PQ5x12np` instead of `PQ5x12`. 

# Implementation-level

There are tradeoffs in terms of code size vs. runtime that can be adjusted at implementation time. 

## C++ templates

Faiss is implemented in C++ and it heavily utilizes template dispatching for various cases. For example, this is the function from [faiss/utils/distances_simd.cpp](https://github.com/facebookresearch/faiss/blob/main/faiss/utils/distances_simd.cpp) that computes inner products between a single `d`-dimenionsal vector `x` and `ny` `d`-dimensional vectors `y`.  
```C++
void fvec_inner_products_ny(
        float* dis,
        const float* x,
        const float* y,
        size_t d,
        size_t ny) {
#define DISPATCH(dval)                                  \
    case dval:                                          \
        fvec_op_ny_D##dval<ElementOpIP>(dis, x, y, ny); \
        return;

    switch (d) {
        DISPATCH(1)
        DISPATCH(2)
        DISPATCH(4)
        DISPATCH(8)
        DISPATCH(12)
        default:
            fvec_inner_products_ny_ref(dis, x, y, d, ny);
            return;
    }
#undef DISPATCH
}
```
As one may see, there are specialized cases for the dimensionalities of 1, 2, 4, 8 and 12, which reference to custom kernels, otherwise the implementation relies on a general-purposes code and a compiler.

Custom specialized kernels tend to show better performance than a general-purpose code, but at the cost of the binary size.

Please feel free to add some custom kernels that fit your case, if appropriate, and send us a pull request with your change! :)

## CUDA templates 

Due to the rigid constraints of GPU computations, templates are more heavily used in the CUDA code than in the C++ code. 

# Feedback

All the feedback regarding the Faiss execution speed is highly appreciated! The most valuable feedback would be a reproducible standalone python script or a small C++ program that exposes performance hot-spots inside Faiss.
