# Overview

Faiss supports indexing binary vectors (with Hamming distance), with the `IndexBinaryFlat`, `IndexBinaryIVF` and `IndexBinaryHNSW` and `IndexBinaryHash/IndexBinaryMultiHash` indexes (all inheriting from `IndexBinary`). 

Those indexes store the vectors as arrays of bytes, so that a vector of size `d` takes only `d / 8` bytes in memory. Note that at the moment, only vectors with sizes multiple of `8` are supported. Of course, you can round up the vector length if needed.

The input to the `add()` and `search()` methods are also byte arrays ("uint8_t" in C++, "uint8" in numpy).
The returned indices are 64-bit ints, distances are 32-bit ints.

From Faiss 1.6.3, most index types also support `range_search`. 

## IndexBinaryFlat

The "flat" binary index performs an exhaustive search. 

The exhaustive search is carefully optimized especially for 256-bit vectors that are quite common. 
The Hamming distance computations are optimized using `popcount` CPU instructions.

Batching is applied on the query and the database side to avoid cache misses. 

The values of `hamming_batch_size` and `faiss::IndexBinaryFlat#query_batch_size` can be customized to adjust the batch sizes but the default values were found to be close to optimal for a large range of settings. 

The examples show how to pass in binary data and how to query the index.

<details><summary>In C++</summary>

```cpp
#include <faiss/IndexBinaryFlat.h>

// Dimension of the vectors, assumed to be a multiple of 8.
int d = 256;

// Vectors to be indexed, each represented by d / 8 bytes, layed out sequentially,
// i.e. the i-th vector starts at db[i * (d / 8)].
int nb = ...;
std::vector<uint8_t> db(nb * (d / 8));
// initialize db

// Vectors to be queried from the index. 
int nq = ...;
std::vector<uint8_t> queries(nq * (d / 8));

// Initializing index.
faiss::IndexBinaryFlat index(d);

// Adding the database vectors.
index.add(nb, db.data());

// Number of nearest neighbors to retrieve per query vector.
int k = ...;

// Output variables for the queries.
std::vector<int32_t> distances(nq * k);
std::vector<faiss::Index::idx_t> labels(nq * k);

// Querying the index
index.search(nq, queries.data(), k, distances.data(), labels.data());

// distances[i * k + j] contains the distance from the i-th query vector to its j-th nearest neighbor.
// labels[i * k + j] contains the id of the j-th nearest neighbor of the i-th query vector.
```
</details>

<details><summary>In Python</summary>

```python
import faiss

# Dimension of the vectors.
d = 256

# Vectors to be indexed, each represented by d / 8 bytes in a nb
# i.e. the i-th vector is db[i].
nb = ...
db = np.empty((nb, d // 8), dtype='uint8')
...initialize db...

# Vectors to be queried from the index.
nq = ...
queries = np.empty((nq, d // 8), dtype='uint8')
...initialize queries...

# Initializing index.
index = faiss.IndexBinaryFlat(d)

# Adding the database vectors.
index.add(db)

# Number of nearest neighbors to retrieve per query vector.
k = ...;

# Querying the index
D, I = index.search(queries, k)

# D[i, j] contains the distance from the i-th query vector to its j-th nearest neighbor.
# I[i, j] contains the id of the j-th nearest neighbor of the i-th query vector.
```
</details>


## IndexBinaryIVF

The "IVF" (Inverse Vector File) flavor speeds up the search by clustering the vectors. This clustering is done using a second (binary) index for quantization (usually a flat index). This is equivalent to the `IndexIVFFlat` of the floating-point indexes.

Examples:

<details><summary>In C++</summary>

```cpp
#include <faiss/IndexBinaryIVF.h>

// Dimension of the vectors.
int d = 256;

// Vectors to be indexed, each represented by d / 8 bytes, layed out sequentially,
// i.e. the i-th vector starts at db[i * (d / 8)].
int nb = ...;
std::vector<uint8_t> db(nb * (d / 8);

// Vectors to train the quantizer.
int nt = ...;
std::vector<uint8_t> training(nt * (d / 8));

// Vectors to be queried from the index.
int nq = ...;
std::vector<uint8_t> queries(nq * (d / 8));

// Initializing the quantizer.
faiss::IndexBinaryFlat quantizer(d);

// Number of clusters.
int nlist = ...;

// Initializing index.
faiss::IndexBinaryIVF index(&quantizer, d, nlist);
index.nprobe = 4; // Number of nearest clusters to be searched per query. 

// Training the quantizer.
index.train(nt, training.data());

// Adding the database vectors.
index.add(nb, db.data());

// Number of nearest neighbors to retrieve per query vector.
int k = ...;

// Output variables for the queries.
std::vector<int32_t> distances(nq * k);
std::vector<faiss::idx_t> labels(nq * k);

// Querying the index
index.search(nq, queries.data(), k, distances.data(), labels.data());

// distances[i * k + j] contains the distance from the i-th query vector to its j-th nearest neighbor.
// labels[i * k + j] contains the id of the j-th nearest neighbor of the i-th query vector.
```
</details>


<details><summary>In Python</summary>

```python
import faiss

# Dimension of the vectors.
d = 256

# Vectors to be indexed, each represented by d / 8 bytes.
# the i-th vector is db[i].
db = ...

# Vectors to train the quantizer.
training = ...

# Vectors to be queried from the index.
queries = ...

# Initializing the quantizer.
quantizer = faiss.IndexBinaryFlat(d)

# Number of clusters.
nlist = ...

# Initializing index.
index = faiss.IndexBinaryIVF(quantizer, d, nlist)
index.nprobe = 4 # Number of nearest clusters to be searched per query. 

# Training the quantizer.
index.train(training)

# Adding the database vectors.
index.add(db)

# Number of nearest neighbors to retrieve per query vector.
k = ...

# Querying the index.
D, I = index.search(queries, k)

# D[i, j] contains the distance from the i-th query vector to its j-th nearest neighbor.
# I[i, j] contains the id of the j-th nearest neighbor of the i-th query vector.
```

</details>

## The `IndexBinaryHNSW`

This is the same method as for the floating point vectors. 

Example usage here: 
[TestHNSW](https://github.com/facebookresearch/faiss/blob/22b7876ef5540b85feee173aa3182a2f37dc98f6/tests/test_index_binary.py#L213)

## The `IndexBinaryHash` and `IndexBinaryMultiHash`

(Faiss 1.6.3 and above)


`IndexBinaryHash`:
A classical method is to extract a hash from the binary vectors and to use that to split the dataset in buckets.
At search time, all hashtable entries within `nflip` Hamming radius of the query vector's hash are visited. 

The hash value is the first `b` bits of the binary vector.
At search time, the number of visited buckets is `1 + b + b * (b - 1) / 2 + .... + comb(b, nflip)`. 

`IndexBinaryMultiHash`:
An extension of this method, studied in [“Fast Search in Hamming Space with Multi-Index Hashing”](http://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf), Norouzi et al, CVPR’12, consists in using `nhash` hash tables built from bits `0:b`, `b:2*b`, … , `(nhash-1) * b: nnhash` * b`.
This is sometimes referred to as LSH because of the multiple hash tables.

A comparison between IndexBinaryHash and IndexBinaryIV is here: [Binary hashing index benchmark](./Binary-hashing-index-benchmark)

# Shorter versions using index factory

The `faiss::index_binary_factory()` allows for shorter declarations of binary indexes. It is especially useful for `IndexBinaryIVF`, for which a quantizer needs to be initialized.

How to use `index_binary_factory`: 

<details><summary>In C++</summary>

Instead of the above initialization code:

```cpp
// Initializing the quantizer.
faiss::IndexBinaryFlat quantizer(d);

// Number of clusters.
int nlist = 32;

// Initializing index.
faiss::IndexBinaryIVF index(&quantizer, d, nlist);
index.nprobe = 4; // Number of nearest clusters to be searched per query. 
```

one could write:

```cpp
#include <faiss/AutoTune.h>
  
// Initializing the quantizer.
faiss::IndexBinaryIVF *index = dynamic_cast<faiss::IndexBinaryIVF *>(faiss::index_binary_factory(d, "BIVF32"));
index->nprobe = 4; // Number of nearest clusters to be searched per query.

```

</details>

<details><summary>In Python</summary>

Instead of the above initialization code:


```python
# Initializing the quantizer.
quantizer = faiss.IndexBinaryFlat(d)

# Number of clusters.
nlist = 32

# Initializing index.
index = faiss.IndexBinaryIVF(quantizer, d, nlist)
index.nprobe = 4 # Number of nearest clusters to be searched per query. 
```

one could write:

```python
# Initializing the quantizer.
index = faiss.index_binary_factory(d, "BIVF32")
index.nprobe = 4 # Number of nearest clusters to be searched per query.

```

</details>

Table of available `index_binary_factory` strings:

| String | Class name | bytes per vector | Comments |
|--------|------------|------------------|--------------| 
| BFlat  | `IndexBinaryFlat` | d/8       | simple flat index |
| BIVF1024 | `IndexBinaryIVF` | d/8 | IVF with 1024 centroids and a flat quantizer |
| BHNSW16 | `IndexBinaryHNSW` | d/8 + 16 * 2 * 4 | HNSW with branching factor M=16.  | 
| BIVF1024_HNSW16 | `IndexBinaryIVF` | d/8 | IVF with 1024 centroids and HNSW M=16 used as a quantizer. |
| BHash32 | `IndexBinaryHash` | d/8 | Binary hash index with 32 bit prefix. |
| BHash4x32 | `IndexBinaryMultiHash` | 4*64/8 | Multiple binary hash index with 32 bits each. |


The bytes per vector are approximate, there is additional memory overhead due to geometric `std::vector` allocation, the coarse quantizer in the IVF and the HSNW tree.

# Asymmetric search with binary codes 

Asymmetric search (ie. database vectors are compressed but the queries are not) is not supported directly with binary indexes. 
Here is a script that does it manually: [demo_asymmetric_binary.ipynb](https://gist.github.com/mdouze/b2e6c6144d4e06fca8287f5257f15fed).