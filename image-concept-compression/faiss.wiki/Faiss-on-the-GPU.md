Some of the most useful indexes are implemented on GPU.

## CPU / GPU interoperability 

The GPU `Index`-es can accommodate both host and device pointers as input to `add()` and `search()`. If the inputs to `add()` and `search()` are already on the same GPU as the index, then no copies are performed and the execution is fastest. Otherwise, a CPU -> GPU copy (or cross-device if the input is resident on a different GPU than the index) will be performed, with a copy back for any results when done (e.g., for `search()`).

The GPU `Index`-es should also be usable as drop-in replacements for anything that expects a CPU `Index`; copies are handled as needed. For example, a `GpuIndexFlatL2` can be used in place of an `IndexFlatL2` and will provide significant acceleration. The `GpuIndexIVFFlat` and `GpuIndexIVFPQ` deliberately do not support the full APIs of their CPU counterparts `IndexIVF`, `IndexIVFFlat` and `IndexIVFPQ` (e.g., direct manipulation of the inverted lists), but most features of the base `Index` API are supported, as well as all of the important indexing features.

### Conversions

Converting from/to GPU is enabled with `index_gpu_to_cpu`, `index_cpu_to_gpu` and `index_cpu_to_gpu_multiple`. 

The two functions that transfer to GPU take an optional `GpuClonerOptions` object, that can be used to adjust the way the GPU stores the objects. The defaults are appropriate for cases where memory is not a constraint. The fields can be adjusted when memory is scarce.

The `GpuClonerOptions` are defined in [gpu/GpuClonerOptions.h](https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/GpuClonerOptions.h). 

### Conversions in Python 

In Python `index_gpu_to_cpu`, `index_cpu_to_gpu` and `index_cpu_to_gpu_multiple` are available. 

For a higher level API without explicit resource allocation, a few easy wrappers [are defined](https://github.com/facebookresearch/faiss/blob/main/faiss/python/gpu_wrappers.py): 

- `index_cpu_to_all_gpus`: clones a CPU index to all available GPUs or to a number of GPUs specified with `ngpu=3`

- `index_cpu_gpu_list`: same, but in addition takes a list of gpu ids that can be specified with `gpus=[2, 4, 6]`

- `index_cpu_to_gpu_multiple_py`: when several GPU indexes are instanciated it is better to factorize the GPU resources to avoid wasting memory. This function takes a list of resource objects that can be re-used between indexes as its first argument, eg.: 

```python
ngpu = 4
resources = [faiss.StandardGpuResources() for i in range(ngpu)]
index1_gpu = faiss.index_cpu_to_gpu_multiple_py(resources, index1)
index2_gpu = faiss.index_cpu_to_gpu_multiple_py(resources, index2)
``` 

All these functions also take a `co=` argument that is a `GpuMultipleClonerOptions` object. 
They can also be used to manage a single GPU just by setting the number of GPUs to 1. 

### Passing in Pytorch tensors 

Pytorch tensors can be input directly to `search()` and `add()`, see [torch_test_contrib_gpu.py](https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/test/torch_test_contrib_gpu.py)

## The GpuResources object

All GPU indexes are built with a `StandardGpuResources` object (which is an implementation of the abstract class `GpuResources`).
The resource object contains needed resources for each GPU in use, including an allocation of temporary scratch space (by default, about 2 GB on a 12 GB GPU), cuBLAS handles and CUDA streams. 

### Scratch memory

The temporary scratch space via the `GpuResources` object is important for speed and to avoid unnecessary GPU/GPU and GPU/CPU synchronizations via `cudaFree`. All faiss GPU code strives to be allocation-free on the GPU, assuming temporary state (intermediate results of calculations and the like) can fit into the scratch space. The temporary space reservation can be adjusted to an amount of GPU memory and even set to 0 bytes via the `setTempMemory` method. 

There are broadly two classes of memory allocations in GPU Faiss: permanent and temporary. Permanent allocations are retained for the lifetime of the index, and are ultimately owned by the index.

Temporary allocations are made out of a memory stack that GpuResources allocates up front, which falls back to the heap (cudaMalloc) when the stack size is exhausted. These allocations do not live beyond the lifetime of a top level call to a Faiss index (or at least, on the GPU they are ordered with respect to the ordering stream, and once all kernels are done on the stream to which all work is ordered, then that temporary allocation is no longer needed and can be reused or freed. Generally about 1 GB or so of memory should be reserved in this stack to avoid cudaMalloc/Free calls during many search operations.

If the scratch memory is too small, you may notice slowdowns due to `cudaMalloc` and `cudaFree`. The high water mark used of the scratch space can be inquired from the resources object, and so can be adjusted to suit actual needs. 

### CUDA streams

All GPU work is ordered with respect to the stream specified in the `GpuResources` object, not necessarily the default (aka null) stream. 
Note that when interfacing with eg. Pytorch without passing data through CPU memory, you should either: 

- force the Faiss `GpuResources` object to use the default stream via `setDefaultNullStreamAllDevices`, or

- add explicit synchronization between the streams


## Implemented indexes 

The index types `IndexFlat`, `IndexIVFFlat`, `IndexIVFScalarQuantizer` and `IndexIVFPQ` are implemented on the GPU, as `GpuIndexFlat`, `GpuIndexIVFFlat`, `GpuIndexIVFScalarQuantizer` and `GpuIndexIVFPQ`. In addition to their normal arguments, they take a resource object as input, along with index storage configuration options and float16/float32 configuration parameters. 

There are 4 different user indices storage options available. User indices are integer values associated with vectors in the database. `INDICES_64_BIT` stores the user-provided indices (for indexed vectors) as a 64 bit signed integer on the GPU itself. `INDICES_32_BIT` stores the indices as 32 bit signed integers on the GPU, for a case where one knows that all indices fit in that range. The end-user API will still upcast the values to `long`. `INDICES_CPU` avoids storing any indices information on the GPU, and instead records it on the CPU. This is useful when GPU space is at a premium, but will involve GPU/CPU copies and lookups. `INDICES_IVF` is useful only for composing a GPU index with other indices that can interpret an inverted file ID and offset.

float16 or float32 precision options affect the storage of data in the database (as with `GpuIndexIVFFlat`), or the storage and processing of intermediate data (as with `GpuIndexIVFPQ`). This option is available on all supported GPUs, from Kepler (K40) to Maxwell and up. float16 is often much more performant and will reduce GPU memory consumption, at the possible cost of accuracy. Recall@N seems mostly unaffected by float16, but estimated distances are affected. GPU architectures with native float16 (aka CUDA `half`) math support, like Pascal, are taken advantage of to provide additional speed. The `GpuIndexFlat` in float16 mode on Pascal will use `Hgemm`; `SgemmEx` is used on other hardware.

### Limitations 

`k` and `nprobe` must be <= 2048 for all indices.

For `GpuIndexIVFPQ`, code sizes per encoded vector allowed are 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64 and 96 bytes.

Memory is generally a scarcer resource on GPU, so there are things to take care of when using the GPU indexes: 

- `GpuIndexIVFPQ` with 56 bytes per code or more requires use of the float16 IVFPQ mode due to shared memory limitations (e.g., 48 x 2^8 x sizeof(float) is 49152 bytes);

- precomputed tables for `GpuIndexIVFPQ` may take up a substantial amount of memory. If you see cudaMalloc errors, disable precomputed tables;

- the indices corresponding to inverted file entries can be stored on CPU rather than on GPU, use `indices_options = INDICES_CPU`

- when memory is really tight, geometric reallocation of the inverted lists can overflow the memory. To avoid this (and generally to increase add speed), call `reserveVecs` on a `GpuIndexIVFPQ` or `GpuIndexIVFFlat` if you know how big the index will be.

- the `StandardGpuResources` object reserves 512 MiB (<= 4 GiB GPUs), 1024 MiB ( <= 8 GiB GPUs), or maximum 1536 MiB (all other GPUs) of GPU memory for temporary calculation space by default. If this is too much, the size can be reduced, at the possible cost of speed.

- adding or searching a large number of vectors should be done in batches. This is not done automatically yet. Typical batch sizes are powers of two around 8192, see [this example](https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_1bn.py#L77-L78).

When converting from a CPU index with `index_cpu_to_gpu`, the default `GpuClonerOptions` object is tuned to maximize speed at the cost of memory usage. 

The disk I/O functions do not support GPU, so a GPU index should be converted to CPU with `index_gpu_to_cpu` before storing it.

### in Python 

In Python, the GPU classes are available if GPU is compiled in.

See [benchs/bench_gpu_sift1m.py](https://github.com/facebookresearch/faiss/blob/master/benchs/bench_gpu_sift1m.py) for a usage example.


## Using multiple GPUs 

Multiple device support can be obtained by:

- copying the dataset over several GPUs and splitting searches over those datasets with an `IndexReplicas`. This is faster (provided queries are provided in large enough batches) because the queries are parallelized. 

- splitting the dataset over the GPUs with an `IndexShards`. This makes it possible to handle larger datasets on GPU.

This is done automatically when transferring an index to GPU with `index_cpu_to_gpu_multiple`. 
By default, it builds an `IndexReplicas`. 
To build an `IndexShards` instead, pass a `GpuMultipleClonerOptions` argument with the field `shard` set to `true`.
For an IVF index, the quantizer can also be factorized by setting the `common_ivf_quantizer` field to `true`.

## Performance

See also [GPU versus CPU](Comparing-GPU-vs-CPU).

GPU faiss varies between 5x - 10x faster than the corresponding CPU implementation on a single GPU (see benchmarks and performance information). If multiple GPUs are available in a machine, near linear speedup over a single GPU (6 - 7x with 8 GPUs) can be obtained by replicating over multiple GPUs. Sharding instead of replication (i.e., each of N GPUs holds 1/N of the database) results in some speedup, but is more sub-linear.

All indices (except for possibly `GpuIndexFlat*`) are mostly memory bandwidth (global or shared) limited rather than arithmetic throughput limited. As with CPU, performance is best with batch queries rather than query size 1. Performance suffers with larger `k` nearest neighbor selection values, especially above 512 or so.

## Supported GPUs 

GPU support is via CUDA. The machine should contain at least one CUDA-capable device of minimum compute capability 3.5 (Kepler and up, K40 included). Warp shuffles (CC 3.0+) and read-only texture caching via `ld.nc`/`__ldg` (CC 3.5+) are the more exotic hardware features used. float16 support requires compiling with CUDA 7.5+.

The GPUs K40, Titan X, M40 and P100 were tested extensively. We have had reports that GTX 970 or a GeForce GTX 1080 crash for some operations (issues [#8](https://github.com/facebookresearch/faiss/issues/8) and [#34](https://github.com/facebookresearch/faiss/issues/34)). We would love to hear feedback (positive or negative) on these platforms.
