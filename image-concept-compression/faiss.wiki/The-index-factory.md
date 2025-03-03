The `index_factory` function interprets a string to produce a composite Faiss index. The string is a comma-separated list of components. 
It is intended to facilitate the construction of index structures, especially if they are nested. 
The `index_factory` argument typically includes a preprocessing component, and inverted file and an encoding component. 
This page summarizes the `index_factory` components and arguments.

A formal definition of the factory strings in Backus–Naur form can be found here: [factory_string.bnf](https://gist.github.com/mdouze/c3111d5f12d1308f5adf78dcd48cdf37).

Examples: 

`index = index_factory(128, "PCA80,Flat")`: produces an index for 128D vectors that reduces them to 80D by PCA then does exhaustive search.

`index = index_factory(128, "OPQ16_64,IMI2x8,PQ8+16")`: takes 128D vectors, applies an OPQ transform to 16 blocks in 64D, uses an inverted multi-index of 2x8 bits (= 65536 inverted lists), and refines with a PQ of size 8, then 16 bytes.

The numbers we indicate are examples, and d is the input dimension. 

## Prefixes 

| String                                     | Class name      | Comments |
|--------------------------------------------|-----------------|----------|
| IDMap                                      | `IndexIDMap`    | Used to enable `add_with_ids` on indexes that do not support it, mainly the flat indexes. |
| IDMap2                                     | `IndexIDMap2`   | Same, but in addition supports `reconstruct`. |



## Vector transforms

These strings map to `VectorTransform` objects that can be applied on vectors before they are indexed

| String | Class name | Output dimension | Comments |
|--------|------------|------------------|----------|
| PCA64, PCAR64, PCAW64, PCAWR64  | `PCAMatrix`  | 64 | Applies a PCA transform to reduce the number of dimensions. W = follow with whitening, R = follow with random rotation. The PCAR is especially useful as pre-processing with a Flat or scalar quantizer. |
| OPQ16, OPQ16_64  |  `OPQMatrix` | d, 64 | Rotates the vectors so that they can be encoded efficiently by a PQ to 16 subvectors. 64 is the output dimension, because it is often beneficial to also reduce the number of dimensions. If the output dimensions not specified, it is the same as the input dimension. |
| RR64             | `RandomRotation` | 64 | Random rotation on the input data. The dimension can increase or decrease wrt. the input |
| L2norm           | `NormalizationTransform` | d | L2-normalizes the input vectors |
| ITQ256, ITQ      | `ITQMatrix`              | 256, d | Applies an ITQ transformation to the input vectors, see ["Iterative quantization: A procrustean approach to learning binary codes for large-scale image retrieval" by Gong et al.](http://slazebni.cs.illinois.edu/publications/ITQ.pdf). This is useful when the vectors are encoded with LSH. |
| Pad128           | `RemapDimensionsTransform` | 128 | Pad the input vectors with 0s to 128 dim |


## Non-exhaustive search components 

### Inverted file indexes

Inverted files all inherit from `IndexIVF`.
The non-exhaustive component specifies what the coarse quantizer (first parameter of the constructor) should be.
The factory strings start with IVF or IMI, followed systematically by a comma and an encoding (see below)

| String | Quantizer class | Number of centroids | Comments |
|--------|------------|----------|-----------|
| IVF4096| `IndexFlatL2` or `IndexFlatIP` | 4096 | Constructs one of the IndexIVF variants, with a flat quantizer. |
| IMI2x9 | `MultiIndexQuantizer` | 2^(2 * 9) = 262144 | Constructs an IVF with many more centroids, possibly less balanced. | 
| IVF65536_HNSW32 | `IndexHNSWFlat` | 65536 | The quantizer is trained as a flat index but indexed with a HNSW. This makes quantization a lot faster | 
| IVF65536(PQ16x4) | arbitrary | 65536 | Use the string in parenthesis to construct the coarse quantizer. | 
| IVF1024(RCQ2x5)  | `ResidualCoarseQuantizer` | 1024 | This is a special case of the previous. The coarse quantizer is a residual quantizer. |

### Graph-based indexes

HNSW and NSG are graph based indexes. They inherit from `IndexHNSW` and `IndexNSG`.
Both rely on a flat storage `IndexFlatCodes` that stores the actual vectors. 

| String | Storage class | Comment | 
|--------|---------------|---------|
| HNSW32, HNSW | `IndexFlatL2` | Arguably the most useful HNSW variant, because when the links are stored, it does not make much sense to compress the vectors. 32 (the number of links per vertex) is the default and can be omitted. | 
| HNSW32_SQ8 | `IndexScalarQuantizer` | SQ8 scalar quantizer | 
| HNSW32_PQ12 | `IndexPQ`             | PQ12x8 index | 
| HNSW32_16384+PQ12 | `Index2Layer`   | 1st layer is a flat index, the PQ encodes the residual of the quantizer | 
| HNSW32_2x10+PQ12 | `Index2Layer`   | 1st layer is an IMI index, the PQ encodes the residual of the quantizer | 

The NSG variants are the same, except that HNSW is replaced with NSG.

### Memory overheads

Memory overheads: 

- In `HNSW32`, 32 encodes the number of links, the lowest level that uses most memory has 32 * 2 links, or 32 * 2 * 4 = 256 bytes per vector. So the overhead is quite significant.

- In `NSG32`, 32 encodes the number of links per vertex directly so 32 means 32 * 4 = 128 bytes per vector.

- For the IVF and IMI indexes, the main overhead is that the 64-bit id of each vector is stored as well (ie. an overhead of 8 bytes per vector)


## Encodings

| String | Class name (Flat/IVF) | code size (bytes) | Comments |
|--------|------------|------------------|--------------| 
| Flat   | `IndexFlat`, `IndexIVFFlat` | 4 * d | The vectors are stored as is, without any encoding |
| PQ16, PQ16x12 | `IndexPQ`, `IndexIVFPQ` | 16, ceil(16 * 12 / 8) | Uses Product Quantization codes with 16 codes of 12 bits each. When the number of bits is omitted, it is set to 8. With suffix "np" does **not** train the Polysemous permutation, which can be slow. |
| PQ28x4fs, PQ28x4fsr, PQ28x4fs_64 | `IndexPQFastScan`, `IndexIVFPQFastScan` | 28 / 2 | Same as PQ above, but uses "fast scan" version of the PQ that relies on SIMD instructions for distance computations. Supports only nbits=4 for now. The suffix _64 indicates the bbs factor used (must be a multiple of 32). The suffix `fsr` (only for IVF) indicates that the vectors should be encoded by residual (slower, more accurate) |
| SQ4, SQ8, SQ6, SQfp16 | `IndexScalar` `Quantizer`, `IndexIVF` `ScalarQuantizer` | `4*d/8`, d, `6*d/8`, `d*2` | Scalar quantizer encoding | 
| Residual128, Residual3x10 | `Index2Layer`  | `ceil(log2(128) / 8)`, `ceil(3*10 / 8)` | Residual encoding. Quantizes the vectors into 128 centroids or 3x10 MI centroids. Should be followed by PQ or SQ to actually encode the residual. Only for use as a codec.  |
| RQ1x16_6x8  | `IndexResidualQuantizer` | (16 + 6*8) / 8 | Residual quantizer codec. The vector is first encoded with a 2^16-centroid quantizer, then the residuals are refined in 6 stages with 256 centroids each.
| RQ5x8_Nqint8  | `IndexResidualQuantizer` | 5+1 | Residual quantizer index with norms quantized to 8 bits (other options are Nqint4 and Nfloat)
| LSQ5x8_Nqint8  | `IndexLocalSearchQuantizer` | 5+1| Same, with local search quantizer 
| PRQ16x4x8 | `IndexProductResidualQuantizer` | 16*4 | The vector is sliced into 16 sub-vectors and each sub-vector is encoded into a residual quantizer of size 4x8.
| PLSQ16x4x8 | `IndexProductResidualQuantizer` | 16*4 | Same, with a LSQ quantizer
| ZnLattice3x10_6 | `IndexLattice` | `ceil((3*log2(C(d/3, 10)) + 6) / 8)` | Lattice codec. The vector is first split into 3 segments, then each segment is encoded as its norm (6 bits) and as a direction on the unit sphere in dimension d/3, quantized by the Zn lattice of radius^2 = 10. `C(dim, r2)` is the number of points on the sphere of radius sqrt(r2) that have integer coordinates (see [here](https://github.com/facebookresearch/spreadingvectors#zn-quantizer) for more details, and [this Gist](https://gist.github.com/mdouze/b167d9a1d0d8838f3427c68c7d412ad8) on how to set the radius).| 
|LSH, LSHrt, LSHr, LSHt | `IndexLSH` | ceil(d / 8) | Binarizes the vectors by thresholding them. At search time, the query vectors are also binarized (symmetric search). Suffix r = rotate vectors prior to binarization, t = train thresholds to balance 0s and 1s. Useful in combination with ITQ. |
| ITQ90,SH2.5 | `IndexIVFSpectralHash` | ceil(90 / 8) | Only an IVF version. Transforms the vector to encode with ITQ (PCA and PCAR also supported), reducing to 90 dim. Then encode each component, with [0, 2.5] giving bit 0 and [2.5, 5] giving bit 1, modulo 5. SH2.5g uses a global threshold, SH2.5c just uses the centroid as threshold. SH without an argument just keeps the sign of the vector components as the bit. |


<!-- 
- with "PQ8+16", works only after an inverted file option, using 8 bytes stored in the inverted index and 16 bytes of refinement. Implemented with `IndexIVFPQR` --> 


## Suffixes

| String | Storage class | Comment | 
|--------|---------------|---------|
| RFlat  | `IndexRefineFlat` | Re-order the search results with exact distance computations |
| Refine(PQ25x12) | `IndexRefine` | Same, but the refine index can be any index. | 

## Example

Let's walk trough one of the most complex factory strings `OPQ16_64,IVF262144(IVF512,PQ32x4fs,RFlat),PQ16x4fsr,Refine(OPQ56_112,PQ56)`: 

- `OPQ16_64`: OPQ pre-processing

- `IVF262144(IVF512,PQ32x4fs,RFlat)`: IVF index with 262k centroids. The coarse quantizer is an IVFPQFastScan index with an additional refinement step.

- `PQ16x4fsr`: the vectors are encoded with PQ fast-scan (which takes 16 * 4 / 8 = 8 bytes per vector)

- `Refine(OPQ56_112,PQ56)`: the re-ranking index is a PQ with OPQ pre-processing that occupies 56 bytes.



