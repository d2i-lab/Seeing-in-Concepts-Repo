Faiss indexes are often composite, which is not easy to manipulate for the individual index types. Also, they have a lot of parameters and it is often difficult to find the optimal structure for a given use case. Therefore, Faiss provides a high-level interface to manipulate indexes in bulk and automatically explore the parameter space.

## I/O and deep copying indexes 

All the functions below produce deep copies, i.e., you don't need to care about object ownership. 

The I/O functions are:

- `write_index(index, "large.index")`: writes the given index to file large.index

- `Index * index = read_index("large.index")`: reads a file

Note that writing GPU indexes is not supported. Please convert to CPU first.

The cloning functions are:

- `Index* index2 = clone_index(index)`: returns a deep copy of the index. Note that some indexes are not supported by the cloning function.

- `Index *index_gpu = index_cpu_to_gpu(resource, dev_no, index)`: deep copies an index, and puts the parts that can be GPU-ified on the GPU. For example, for an `IndexPreCompute` that includes an `IndexIVFPQ`, only the `IndexIVFPQ` will be copied to GPU. A 4th argument can be provided to set the copying options (float16 precision and such), see [Faiss on the GPU](./Faiss-on-the-GPU). 

- `Index *index_cpu = index_gpu_to_cpu(index)`: copy in the other direction

- `index_cpu_to_gpu_multiple`: uses an `IndexShards` or `IndexReplicas` to copy the index to several GPUs. 

> __Warning__
No attempt is made to check the that loaded data is correct. Therefore, a faulty or malicious file that is loaded via these functions could trigger out-of-memory errors or when crafted expertly code execution. Therefore, it is the user's responsibility to verify that a file that was stored by the `write_index` function was not altered before reading with `read_index`. 

### Generic I/O support 

The basic I/O functions are for files. 
However, Faiss indexes can be serialized to any channel that supports a write operation for chunks of binary data (or read for reading). 
Random access to the channel is not required.

In the C++ API, this appears as the `IOReader` and `IOWriter` classes, see [faiss/impl/io.h](https://github.com/facebookresearch/faiss/blob/master/faiss/impl/io.h). 

The only operation that these classes expose is `operator ()`, which works similarly as stdio's `fwrite` and `fread`.
In Python, these operations are exposed as `read_bytes` and `write_bytes`.

### Standard I/O object implementations

- `FileIOReader` and `FileIOWriter`: the object that are used to write to files. Note that memory mapping is supported only for these types of indexes. 

- `VectorIOReader` and `VectorIOWriter` read/write to a `std::vector<uint8_t>`. 
These are exposed in the Python functions `serialize_index` and `deserialize_index`, see [python/faiss.py](https://github.com/facebookresearch/faiss/blob/a17a631/python/faiss.py#L756), that serialize indexes to numpy uint8 arrays.

- `BufferedIOReader` and `BufferedIOWriter`: wrap another index to add a buffering layer and avoid too small reads or writes. 

### I/O with Python callbacks

To support I/O types that exist only in python, eg. file-like objects with a read()/write() function, use the `PyCallbackIOReader` (and `PyCallbackIOWriter`). 
They just take a function callback (can be an object method) that reads from or writes a `bytes` object (or `str` in Python2).
Example here: [tests/test_io.py](https://github.com/facebookresearch/faiss/blob/a17a631/tests/test_io.py#L125).

## Auto-tuning the runtime parameters 

In this section we focus on a subset of the Faiss indexes, that are selected because they are most useful to index 1M to 1G vectors under various constraints.

The parameters of the indexes can be classified in: 

- build-time parameters that must be set when the index is built

- run-time parameters that can be adjusted before performing a search

Auto-tuning is performed on the runtime parameters. The parameters that are adjusted are:

| index type | Index class | runtime parameter  | comments 
| ----|-------|--------------------|----------
| IVF*, IMI2x* | IndexIVF* | nprobe | the main parameter to adjust the speed-precision tradeoff
| IMI2x* | IndexIVF | max_codes | useful for the IMI, that often has unbalanced inverted lists
| HNSW* | IndexHNSW | efSearch | the depth of the HNSW search 
| PQ* | IndexIVFPQ, IndexPQ | ht | Hamming threshold for polysemous 
| PQ*+* | IndexIVFPQR | k_factor | determines how many result vectors are verified
| Refine | IndexRefine | kf_factor | determines how many vectors are verified per queried result

The auto-tuning explores the speed-accuracy space and keeps the Pareto-optimal points in that space. 
When a parameters applies to a coarse quantizer in an IVF index, it is prefixed by `quantizer_`. 
For example for an `IVF_HSNW32,Flat` index, the HNSW efSearch parameter can be set with `quantizer_efSearch`.

### The `AutoTuneCriterion` object

The `AutoTuneCriterion` contains ground-truth results for the search and evaluates search results. It returns a performance measure between 0 and 1. Currently there are implementations for 1-recall@R and R-recall@R (aka. intersection) criterions.

### The OperatingPoints object

The object stores all operating points as (performance, search time, parameter_set_id) tuples and selects the operating points that are optimal. An operating point is optimal if there is no other operating point that is faster to get at least the same performance. 

### The ParameterSpace object 

The ParameterSpace object scans a given index for tunable parameters and builds a table of possible values for each parameter. 

The parameter space is exponentially large with the number of parameters. We explore it in a random order.

We apply a heuristic to prune the exploration space. For each parameter, a higher value is slower and more accurate. Therefore, given two tuples of parameters a, b with their partial order, if 

  a >= b 

then a is slower and more accurate than b. Therefore, before evaluating a set of parameters a we check if there exists (b, c) st. a >= b and c >= a. If the (perf_c, time_b) is not an optimal operating point, then we skip the experiment because it is guaranteed to produce a non-optimal operating point. This quickly prunes the set of experiments to realize.

A set of parameters is expressed as a human-readable string ("nprobe=2,ht=52") and an index (cno).

### ParameterSpace as a way to set parameters on an opaque index

The `ParameterSpace` object can set run-time parameters on an index. 
A typical usage is (in C++, python is equivalent): 

```
Index *index = ....;
ParameterSpace().set_index_parameters(index, "nprobe=5,quantizer_efSearch=8");
```

The parameters are separated with a comma. 
If a parameter is not recognized, `set_index_parameters` throws an exception.


## Example usage

See [`faiss/demos/demo_sift1M.cpp`](https://github.com/facebookresearch/faiss/blob/main/demos/demo_sift1M.cpp) for an example usage of auto-tuning. The output reads like: 

```
  0/500: cno=0 nprobe=1,ht=2 bounds [perf<=1.000 t>=0.000]  perf 0.000 t 0.061
  1/500: cno=779 nprobe=2048,ht=256 bounds [perf<=1.000 t>=0.061]  perf 0.627 t 16.697 *
  ...
  7/500: cno=247 nprobe=128,ht=42 bounds [perf<=0.000 t>=0.270] skip
  8/500: cno=664 nprobe=16,ht=112 bounds [perf<=0.282 t>=0.064]  perf 0.017 t 0.150 *
  9/500: cno=555 nprobe=8,ht=94 bounds [perf<=0.017 t>=0.064]  perf 0.000 t 0.092
  10/500: cno=292 nprobe=16,ht=50 bounds [perf<=0.000 t>=0.061] skip
  ...
```

This means that experiment 8 tests parameter combination 664 which sets nprobe to 16 and the Hamming threshold to 112. Previous experiments show that the performance is below 0.282 and the runtime above 0.064 s. The result performance (1-recall at 1 in this case) is 0.017 obtained in 0.150s. The star at the end of the line indicates that the operating point is optimal in the set of resutls seen so far.

The output result is the list of operating points that are optimal:

```
cno=720 key=nprobe=1,ht=122 perf=0.0819 t=0.046
cno=710 key=nprobe=4,ht=120 perf=0.0895 t=0.059
cno=732 key=nprobe=1,ht=124 perf=0.1139 t=0.068
cno=744 key=nprobe=1,ht=126 perf=0.1488 t=0.069
cno=734 key=nprobe=4,ht=124 perf=0.1775 t=0.080
cno=746 key=nprobe=4,ht=126 perf=0.2264 t=0.089
cno=770 key=nprobe=4,ht=256 perf=0.5010 t=0.095
cno=771 key=nprobe=8,ht=256 perf=0.5564 t=0.156
cno=772 key=nprobe=16,ht=256 perf=0.5936 t=0.197
cno=774 key=nprobe=64,ht=256 perf=0.6226 t=0.626
cno=776 key=nprobe=256,ht=256 perf=0.6267 t=2.246
cno=778 key=nprobe=1024,ht=256 perf=0.6268 t=8.601
```

The string `"nprobe=4,ht=126"` can be used to set the parameters on the index. 

## Reliability of the optimization 

There are 3 parameters that impact the reliability of auto-tuning (or manual tuning, btw):

- sensitivity of the query set and the criterion. If it is too easy (eg. R@1 on Twitter Glove) or too hard (eg. the random dataset), no useful signal will come from the tuning. 

- the dataset should have enough query points, at least 1000, preferably 10000, to be sensitive enough to the parameter settings.

- reliability of the timings. Times are measured in wall-clock time. This is most reliable on an unoccupied GPU or 1 thread of CPU. It is less reliable in multi-threading. The smaller the dataset, the less reliable the timing. Multi-thread measurements below 1s are not reliable.



