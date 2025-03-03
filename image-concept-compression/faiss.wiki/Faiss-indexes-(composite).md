This page presents more advanced features of Faiss indexes. The best operating points can be obtained by combining several of the indexing methods described in the previous section.

## Cell probe method with a PQ index as coarse quantizer

A product quantizer can also be used as a coarse quantizer. This corresponds to the Multi-Index described in [The inverted multi-index, Babenko & Lempitsky, CVPR'12]. For a PQ with m segments each encoded as c centroids, the number of inverted lists is c^m. Therefore, m=2 is the only practical option.

In FAISS, the corresponding coarse quantizer index is the `MultiIndexQuantizer`. This index is special because no vector is added to it. Therefore a specific flag (`quantizer_trains_alone`) has to be set on the `IndexIVF`.

```python
nbits_mi = 12  # c
M_mi = 2       # m
coarse_quantizer_mi = faiss.MultiIndexQuantizer(d, M_mi, nbits_mi)
ncentroids_mi = 2 ** (M_mi * nbits_mi)

index = faiss.IndexIVFFlat(coarse_quantizer_mi, d, ncentroids_mi)
index.nprobe = 2048
index.quantizer_trains_alone = True
```

The `MultiIndexQuantizer` typically is competitive compared to an `IndexFlat` in fast/low-precision operating points.


## Pre-filtering PQ codes with polysemous codes

It is about 6x faster to compare codes with Hamming distances than to use a product quantizer. However, by a proper reordering of the quantization centroids, the Hamming distances between PQ codes become correlated with the true distances. The by applying a threshold on the Hamming distance, most expensive PQ code comparisons can be avoided.

To enable this on an `IndexPQ`:

```python
index = faiss.IndexPQ (d, 16, 8)
# before training
index.do_polysemous_training = true
index.train (...)

# before searching
index.search_type = faiss.IndexPQ.ST_polysemous
index.polysemous_ht = 54    # the Hamming threshold
index.search (...)
```

For an `IndexIVFPQ`:

```python
index = faiss.IndexIVFPQ (coarse_quantizer, d, 16, 8)
# before training
index. do_polysemous_training = true
index.train (...)

# before searching
index.polysemous_ht = 54 # the Hamming threshold
index.search (...)
```

To set a reasonable threshold, keep in mind that:

- the threshold should be between 0 and the number of bits per code (128 = 16*8 in this case), and  codes follow a binomial distribution

- setting the threshold to 1/2 the number of bits per code will spare 1/2 of the code comparisons, which is not enough. It should be set to a lower value (hence the 54 for 128 bit codes).

## IndexIVFPQR: refining IVFPQ search results with an additional product quantizer

The `IndexIVFPQR` adds an additional level of quantization (the third!) on top of an IndexIVFPQ. Similar to the IndexRefineFlat It refines the distances computed by an IndexIVFPQ and reorders the results based on these.

## Whole class hierarchy 

For the curious, the CPU Faiss class hierarchy looks like this [faiss_class_hierarchy.pdf](./faiss_class_hierarchy.pdf)