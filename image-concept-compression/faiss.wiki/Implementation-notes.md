Here are a few notes on implementation details for which we sometimes get questions. 
We describe the tradeoffs and maybe a few unexpected design choices or results. 

# Matrix multiplication to do many L2 distance computations

A typical operation in `IndexFlatL2` is to exhaustively compare a set of `nq` query vectors and a set of `nb` database vectors in dimension d (then select the top-k smallest vectors).

Faiss has two implementations of this operation: 

1. direct implementation that loops over nq, nb and the dimension of the vectors. 

2. an implementation that uses the decomposition $||x -  y||^2 = ||x||^2 + ||y||^2 - 2 \left< x, y \right>$. 
This is faster because the most expensive operation in O(nq * nb * d) can be handed over to BLAS that normally does this efficiently. 

We use implementation 1 when nq < 20, and implementation 2 otherwise. 
The threshold 20 can be adjusted via global variable `faiss::distance_compute_blas_threshold` (accessible in Python via `faiss.cvar.distance_compute_blas_threshold`). 

Note that solution 2 may be less stable numerically than 1 for vectors of very different magnitudes, see discussion in issue [#297](https://github.com/facebookresearch/faiss/issues/297).

# k-means implementation

k-means is implemented in the [Clustering object](https://github.com/facebookresearch/faiss/blob/master/Clustering.h)

After initialization, the k-means iterates two operations: 
 
- assign training points to centroids

- recompute centroids as the center of mass of the points they are assigned to.

In terms of performance, the first operation is the most costly (by far). 
Incidentally, it can be performed by any index, since it is a nearest-neighbor search of the vectors to the centroids. 
Therefore the index is a parameter of the Clustering train method.
It can be replaced with a GPU index ([example](https://github.com/facebookresearch/faiss/blob/master/benchs/kmeans_mnist.py#L45)) or a HNSW index ([example](https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py#L172)). 

# Precomputed tables in IVFPQ 

PQ search boils down to doing look-ups in distance tables (as explained in the original paper [“Product quantization for nearest neighbor search”](https://hal.inria.fr/inria-00514462v2/document)). 

For the IVFPQ, an additional level of pre-computation can be done to speed up the construction of these tables. 
This is explained in ["Billion-scale similarity search with GPUs"](https://arxiv.org/abs/1702.08734), section 5.2. 
It is necessary only for L2 distance computations and residual encoding (which is the default).
There is a tradeoff between memory usage to store the additional tables and speed: 

- on CPU, it is governed by `IndexIVFPQ.use_precomputed_table`. It takes 4 possible values: -1=disable, 0=decide heuristically (default: use tables only if they are < `IndexIVFPQ::precomputed_tables_max_bytes`, set to 2GB by default); 1=enable (size 256 * nlist * M); 2=specific version for the `MultiIndexQuantizer` that is much more compact. 
Calling `precompute_table()` takes `use_precomputed_table` into account and updates the data for the next search.

- on GPU, precomputed tables are enabled if the `GpuIndexIVFConfig::usePrecomputedTables` is set to true at construction. 
The setting can be changed at runtime with `GpuIndexIVF::setPrecomputedCodes(bool enable)`. 

# Transposed centroids table for PQ

Trained `ProductQuantizer` struct maintains a list of centroids in an 1D array field called `::centroids`, its layout is `(M, ksub, dsub)`. This allows to access the coordinates of the centroids directly.

Faiss 1.7.3 introduces two new fields, which allow to perform the calls to `ProductQuantizer::compute_code()` faster:
- `::transposed_centroids` which stores the coordinates of the centroids, its layout is `(dsub, M, ksub)`. 
- `::centroids_sq_lengths` which stores the squared lengths of the centroids, its layout is `(M, ksub)`.

These fields are not supposed to be accessed directly and are treated as temporary ones.

Instead, 
1. Prepare a ProductQuantizer
2. Edit the coordinates of the centroids by training the product quantizer or accessing `::centroids` field, if needed.
3. Use a new function `ProductQuantizer::sync_transposed_centroids()` which automatically populates these two fields based on the values in `::centroids` field. As a result, a different, more computationally efficient kernel will be used to evaluate `ProductQuantizer::compute_code()` call. Currently, the kernel is available for `dsub`=1, 2, 4 and 8.
4. Perform `ProductQuantizer::compute_code()` or `ProductQuantizer::compute_codes()`.
5. Use `ProductQuantizer::clear_transposed_centroids()` to clear this buffer, if it is not needed any more.

The code snippet for ProductQuantizer is the following:
```python
# prepare pq
pq = faiss.ProductQuantizer(...)
pq.train(...)

# enable transposed centroids table to speedup compute_codes()
pq.sync_transposed_centroids()

# compute    
codes_transposed = pq.compute_codes(...)

# disable transposed centroids table once it is no longer needed
pq.clear_transposed_centroids()
```

Please note that `pq.compute_codes()` might produce a bit different results if transposed centroids tables are used (compared to the case if ones are not used) because of the finite precision of floating point values.

# PCA matrix computation

The PCA matrix is computed by: 

- if there are more training points than dimensions: compute the covariance matrix, then use the LAPACK dsyev to extract the eigenvalues and vectors.

- if there are more dimensions than training points: compute the Gram matrix (pairwise dot products of training vectors), and extract the eigenvalues and vectors from that. 

# Statistics for non-exhaustive search

Non-exhaustive search means that only part of the database vectors are compared with. 
This is the case with `IndexIVF` variants and `IndexHSNW` variants. 

The number of distances that are effectively computed are collected in global variables. 
For `IndexIVF` this is `indexIVF_stats.ndis`, for `HNSW` it is `hnsw_stats.ndis`. 
These statistics are guaranteed to be accurate only if the `search` function is not called from multiple threads.

To access the variables in Python, just prefix with `faiss.cvar`, eg. `faiss.cvar.indexIVF_stats.ndis` will contain the number of distances computed so far.

# The 4-bit PQ fast-scan implementation in Faiss

See also [Fast accumulation of PQ and AQ codes (FastScan)](https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan))

Most product quantizer (PQ) decompositions use 8 bits per sub-vector, which is convenient because it is byte-aligned. 
At search time, the look-up tables are stored in RAM (hopefully in cache) and can be used for distance computations, see equation (13) in  [Product quantization for nearest neighbor search](https://hal.inria.fr/inria-00514462/document). 

However, modern CPUs are more efficient when computing arithmetic operations than doing memory look-ups. 
One way to avoid this is to degrade to less powerful vector encodings (like binary or scalar decompositions). 

Another way is to store the search-time look-up tables in registers. 
This idea was introduced in ["Cache locality is not enough: high-performance nearest neighbor search with product quantization fast scan", André et al, VLDB 15](http://www.vldb.org/pvldb/vol9/p288-andre.pdf), hence the name "fast-scan" of the current implementation.

Fitting look-up tables in registers is possible only under the following conditions: 

- there must be many vector registers and there must be a look-up instruction (aka. shuffle). 
This is supported on architectures like the Intel AVX (16 256-bit registers) or ARM Neon (32 128-bit registers).

- the LUT tables must be short: 8-bit PQ produces 256 LUT entries, so reduce it to 4-bit. 

- the LUT table entries must be small: we cannot afford to store floating point entries, so they are quantized to 8 bit per entry.


## Implementation

The 4-bit PQ fast-scan implementation is heavily inspired by the [Google SCANN](https://github.com/google-research/google-research/tree/master/scann). 

It supports only AVX2 currently. 
Compiling without AVX2 works but is very inefficient because the vector registers are emulated in memory. 

The `IndexPQFastScan` and `IndexIVFPQFastScan` objects implement PQ4 fast scan. 

The main difference with the regular `IndexPQ` and `IndexIVFPQ` is that the codes are laid out by batches of 32 (64 and 96 are supported as well but there are few operating points where they are competitive).


# Reranking

Since the 4-bit PQ has a relatively low accuracy (PQ32x4 is much less accurate than PQ16x8 although they use the same amount of memory), it is useful to perform a re-ranking stage. 
For example, if we need k=10 results, we instead query k * k_factor_rf = 100 elements and rerank the top-100 resutls per query with exact (or more accurate) distance computations. 

This 2-stage encoding is used in several works, including SCANN, the 3-level PQ search in [Searching in one billion vectors: Re-rank with source coding
, Jegou et al, ICASSP'11](https://arxiv.org/pdf/1102.3828.pdf), [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node, Subramanya et al. NeurIPS'19](http://harsha-simhadri.org/pubs/DiskANN19.pdf). 

In Faiss, this is supported via the `IndexRefine` object. 
It manages two indexes, the second one being refining the results from the first index. 

## Parallel sorting functions 

For some Faiss use cases, having efficient implementations for very basic functions is critical. 

[faiss/utils/sorting.h](https://github.com/facebookresearch/faiss/blob/main/faiss/utils/sorting.h) exposes efficient 
sorting functions: 

- `fvec_argsort_parallel` is a parallel argsort implementation

- `bucket_sort` is a bucket sort function, ie it shuffles the input array (indirectly, with a permutation) so that the array entries 0 come first, then 1, ..., then `nbucket-1`. It is exposed in Python as [`faiss.bucket_sort`](https://github.com/facebookresearch/faiss/blob/main/faiss/python/extra_wrappers.py#L146)

- `matrix_bucket_sort_inplace` is a bit more complicated: it sorts elements inplace and in addition records from which row each elements comes. In Python: [`faiss.matrix_bucket_sort`](https://github.com/facebookresearch/faiss/blob/main/faiss/python/extra_wrappers.py#L179)

- `hashtable_int64_to_int64_init`, `..._add`, `..._lookup`: a parallel hashtable implementation. Allows to add and search batches of (key, value) pairs (both int64) to a hashtable stored in an external table. Exposed in Python as [`MapInt64ToInt64`](https://github.com/facebookresearch/faiss/blob/main/faiss/python/extra_wrappers.py#L304) 

# Memory layout of multiple quantization indices

The product quantizer and additive quantizers produce a series of $M$ $n$-bit quantization indices (noted as PQ$M\times n$): $q_1,\dots,q_n$.

## Per byte layout

When $n=8$, the memory layout of these indices is quite natural: the first byte contains $q_1$, the second $q_2$, etc. 

When $n \ne 8$, the bits are filled in starting from the least significant bit of each byte. 
For example, for $n=6$: 

- the 6 low bits of byte 1 contain $q_1$

- the 2 high bits of byte 1 contain the 2 low bits of $q_2$

- the 4 low bits of byte 2 contain the 4 high bits of $q_2$

etc. 

## Loading as 64-bit integers

This memory representation is sometimes used as a single integer. 
This is useful in particular when the whole bit string is used as a coarse quantizer index, eg. in `MulitIndexCoarseQuantizer` or `AdditiveCoarseQuantizer`. 

What's convenient is that on a [little endian machine](https://en.wikipedia.org/wiki/Endianness) (Intel and ARM machines are little endian), this means that the bits are populated from lowest to highest. 
In the previous example, with $n=6$, stored in `q`:

- $q_1$ is stored int the 6 lowest bits of `q`: $q_1$=`q & 0x3f`

- $q_2$ is stored int the bits 6..11 of `q`: $q_2$=`(q >> 6) & 0x3f`

etc.

A general way of reading these bits is via the `BitstringReader` object, see [this code](https://github.com/facebookresearch/faiss/issues/2285#issuecomment-1087228404).

## Dispatching runtime parameters to fixed parameters

Often the compiler can optimize more agressively when a parameter is known at compile time.
Then a function can be converted into a _kernel_, with few tests and runtime dependencies. 
This is typically obtained by converting a discrete parameter into a template parameter of the function. 
Then the function can be compiled several times for difference settings of the parameter. 

The difficulty is to dispatch the runtime parameter into the kernel implementations. 
This is typically obtained via `switch`/`case` statements. 
However, when multiple functions must be specialized with the same dispatching code, this means the switch/cases must be repeated. 
To avoid this, we use a nifty C++ functionality: variadic template parameters. 

For example, [this piece of code](https://github.com/facebookresearch/faiss/blob/3fe0b934f3c0b3708e28a3879ce02b81a1ede53f/faiss/utils/hamming_distance/hamdis-inl.h#L39) defines several implementations of Hamming distance computations for some common sizes of binary signatures. 

To dispatch the size of the signature to the relevant kernel, we use [dispatch_HammingComputer](https://github.com/facebookresearch/faiss/blob/3fe0b934f3c0b3708e28a3879ce02b81a1ede53f/faiss/utils/hamming_distance/hamdis-inl.h#L55). 
Dispatcher functions have the general format: 
```c++ 

template <class Consumer, class... Types>
typename Consumer::T dispatch_HammingComputer(
        int code_size,   //< parameters necessary to build the class that Consumer::f is templatized on 
        Consumer& consumer,
        Types... args) {
    switch (code_size) {
       /// this is typically a macro because it contains lots of repeated code
       /// it construct the template parameter for Consumer::f
#define DISPATCH_HC(CODE_SIZE) \
    case CODE_SIZE:            \
        return consumer.template f<HammingComputer##CODE_SIZE>(args...); 

        /// handle the different cases
        DISPATCH_HC(4);
        DISPATCH_HC(8);
        DISPATCH_HC(16);
        DISPATCH_HC(20);
        DISPATCH_HC(32);
        DISPATCH_HC(64);
        default:
            return consumer.template f<HammingComputerDefault>(args...);
    }
}

```

The dispatcher function is then called with a `Consumer` class as template parameter (unfortunately it does not seem to be possible with a function directly). 
See [here](https://github.com/facebookresearch/faiss/blob/3fe0b934f3c0b3708e28a3879ce02b81a1ede53f/faiss/IndexBinaryHash.cpp#L179) for an example. 
Explanation here: 
```c++
// object that defines T and f and forwards the parameters to the templatized function. 
// the function could also be defined in f's body
struct Run_search_single_query {
    using T = void;
    template <class HammingComputer, class... Types>
    T f(Types... args) {
        search_single_query_template<HammingComputer>(args...);
    }
};

void search_single_query(
        const IndexBinaryHash& index,
        const uint8_t* q,
        SearchResults& res,
        size_t& n0,
        size_t& nlist,
        size_t& ndis) {
    Run_search_single_query r; // instanciate consumer object 
    dispatch_HammingComputer(
            index.code_size, // parameter(s) passed to the dispatcher
            r, // consumer 
            index, q, res, n0, nlist, ndis // extra parameters for Consumer::f
    );
}

```

Note that there is a non-negligible risk of code bloat by making dispatching so easy. 
In general it's healthy to use common sense and/or benchmarking to verify that freezing the parameters does indeed speed up something.