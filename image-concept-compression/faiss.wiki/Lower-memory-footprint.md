## This uses too much memory, how can I shrink the storage?

The indexes we have seen, `IndexFlatL2` and `IndexIVFFlat` both store the full vectors. To scale up to very large datasets, Faiss offers variants that compress the stored vectors with a lossy compression based on product quantizers.

The vectors are still stored in Voronoi cells, but their size is reduced to a configurable number of bytes m (d must be a multiple of m).

The compression is based on a [Product Quantizer](https://hal.archives-ouvertes.fr/file/index/docid/514462/filename/paper_hal.pdf), that can be seen as an additional level of quantization, that is applied on sub-vectors of the vectors to encode.

In this case, since the vectors are not stored exactly, the distances that are returned by the search method are also approximations.

### In Python

```python
nlist = 100
m = 8                             # number of subquantizers
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])
```

<!--
### In Lua

```lua
  nlist = 100
  m = 8                             -- number of subquantizers
  k = 4
  quantizer = faiss.IndexFlatL2(d)  -- this remains the same
  index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    -- 8 specifies that each sub-vector is encoded as 8 bits
  index:train(xb)
  index:add(xb)
  D, I = index:search(xb:sub(1, 5), k) -- sanity check
  print(I)
  print(D)
  index.nprobe = 10              -- make comparable with experiment above
  D, I = index:search(xq, k)     -- search
  print(I:sub(-5, -1))
```
-->
### In C++

```c++
    int nlist = 100;
    int k = 4;
    int m = 8;                             // number of subquantizers
    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);

    index.train(nb, xb);
    index.add(nb, xb);
    {       // sanity check
        ...
        index.search(5, xb, k, D, I);
        printf("I=\n");
        ...
        printf("D=\n");
        ...
    }
    {       // search xq
        ...
        index.nprobe = 10;
        index.search(nq, xq, k, D, I);
        printf("I=\n");
        ...
    }
```

### Results

The results look like:

```
[[   0  608  220  228]
 [   1 1063  277  617]
 [   2   46  114  304]
 [   3  791  527  316]
 [   4  159  288  393]]

[[ 1.40704751  6.19361687  6.34912491  6.35771513]
 [ 1.49901485  5.66632462  5.94188499  6.29570007]
 [ 1.63260388  6.04126883  6.18447495  6.26815748]
 [ 1.5356375   6.33165455  6.64519501  6.86594009]
 [ 1.46203303  6.5022912   6.62621975  6.63154221]]
```

We can observe that the nearest neighbor is found correctly (it is the vector ID itself), but the estimated distance of the vector to itself is not 0, although it is significantly lower than the distance to the other neighbors. This is due to the lossy compression.

Here we compress 64 32-bit floats to 8 bytes, so the compression factor is 32.

When searching on real queries, the results look like:

```
[[ 9432  9649  9900 10287]
 [10229 10403  9829  9740]
 [10847 10824  9787 10089]
 [11268 10935 10260 10571]
 [ 9582 10304  9616  9850]]
```

They can be compared with the `IVFFlat` results above. For this case, most results are wrong, but they are in the correct area of the space, as shown by the IDs around 10000. The situation is better for real data because:

- uniform data is very difficult to index because there is no regularity that can be exploited to cluster  or reduce dimensionality

- for natural data, the semantic nearest neighbor is often significantly closer than irrelevant results.

### Simplifying index construction

Since building indexes can become complicated, there is a factory function that constructs them given a string. The indexes above can be obtained with the following shorthand:

```python
index = faiss.index_factory(d, "IVF100,PQ8")
```

```c++
faiss::Index *index = faiss::index_factory(d, "IVF100,PQ8");
```
Replace `PQ8` with `Flat` to get an `IndexFlat`. The factory is particularly useful when preprocessing (PCA) is applied to the input vectors. For example, the factory string to preprocess reduce the vectors to 32D by PCA projection is: `"PCA32,IVF100,Flat"`. 

## Further reading

Explore the next sections to get more specific information about the types of indexes, GPU faiss, coding structure, etc.