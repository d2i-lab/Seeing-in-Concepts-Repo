## GPU versus CPU: performance

The GPU usually represents a significant speedup over the CPU implementations, but there are two issues to keep in mind:

- **Overhead of CPU <-> GPU copies**

  Typically, the CPU is connected to the GPU over a bus with lower bandwidth than that of the CPU to its main memory, and especially the CPU to its own caches; e.g., PCIe3 will max out at about 12 GB/sec, while server-class CPUs typically have 50+ GB/sec of total all-core cross-sectional memory bandwidth.

  Copying a populated index from the CPU to the GPU may take a lot of time. Only if you are performing a reasonable number of queries on the GPU will you amortize this overhead; it is best to copy an index once to a GPU and keep it there, or to create an index on the GPU and populate it there.

  If you have a lot of query vectors on the CPU, copying this to the GPU could take some time as well, though this is only an issue if the index is small (think: the index on CPU could fit in a CPU's last level cache, on the order of a few MB).

- **Batch size and index size**

  GPUs are typically higher latency but have higher parallel throughput and memory bandwidth than CPUs. It is best to use batch queries with the CPU or GPU if possible as this amortizes the touching of index memory across all of the queries.

  Index size should be relatively large to see the GPU win as well, then it will win massively. An index with only a few thousand vectors will typically always be faster on the CPU (as it can fit in the CPU's caches), but hundreds of thousands to millions/billions of vectors will be great to amortize the GPU's overheads.

  To summarize:

  - small query batch, small index: CPU is typically faster
  - small query batch, large index: GPU is typically faster
  - large query batch, small index: could go either way
  - large query batch, large index: GPU is typically faster

The GPU indices support query data coming from the host CPU or from GPU memory. If the query data is already on the GPU, then using it on the GPU would be best, making the GPU strategy win in all 4 cases except for the most degenerate (super tiny index with little parallelism available for the GPU).

## GPU versus CPU: accuracy

The GPU indices implement the same algorithms as the CPU, but may or may not return the exact same results. there are three main things to keep in mind:

- **floating point reduction order**
  
  Floating point arithmetic is not associative by default. The order of operations in the GPU code is liable to be quite different than the CPU version, so the final reported distance results or order of elements returned could differ from those reported by the GPU. Even in the simplest case of a CPU `IndexFlatIP` versus GPU `GpuIndexFlatIP` could report different results due to floating point reduction differences in the respective matrix multiplication kernels.

- **equivalent element k-selection order**

  The order in which results are scanned and k-selected on the GPU is not guaranteed to be the same as the CPU. If you have equivalent values (i.e., as vectors in the index, or distances reported), there is no guaranteed relative order of these equivalent elements, similar to a [non-guarantee of stability for a sort](https://en.wikipedia.org/wiki/Sorting_algorithm#Stability).

  For example, an index could have 1000 duplicate vectors, each with different user IDs. If some of these vectors are within min-k (L2) or max-k (IP), the GPU may return different IDs than the CPU.

- **float16 opt-in**

  If float16 versions of GPU algorithms are used (e.g., for `GpuIndexFlat`), then distance calculations will be different as well.

In order to compare CPU to GPU equivalency, one should probably use a recall @ N framework to determine the level of overlap between the CPU and GPU results, and for results with the same ID between GPU and CPU, the distances should be within some reasonable epsilon (say, 1-500? [units in the last place](https://en.wikipedia.org/wiki/Unit_in_the_last_place)).
