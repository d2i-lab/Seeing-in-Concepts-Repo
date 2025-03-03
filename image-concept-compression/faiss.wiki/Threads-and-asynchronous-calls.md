About threading in Faiss.

## Thread safety

Faiss CPU indices are thread-safe for concurrent searches, and other operations that do not change the index.
A multithreaded use of functions that change the index needs to implement mutual exclusion.

Faiss GPU indices are not thread-safe, even for read only functions. This is because 
`StandardGpuResources` for GPU Faiss is not thread-safe, as it manages a region of temporary memory on the GPU which can only have a single user at a time. A single `StandardGpuResources` object must be created for each CPU thread that is actively running a GPU Faiss index. Multiple GPU indices managed by a single CPU thread and share the same `StandardGpuResources` (and indeed should, as they can use the same temporary regions of GPU memory). A single `GpuResources` object can support multiple devices, but only from a single calling CPU thread.
Multi-GPU Faiss (obtained via `index_cpu_to_gpu_multiple`) does internally run different GPU indices from different threads.

## Internal threading 

Faiss itself is internally threaded in a couple of different ways. For CPU Faiss, the three basic operations on indexes (training, adding, searching) are internally multithreaded. Threading is done through OpenMP, and a multithreaded BLAS implementation. Faiss does not set the number of threads. The caller can adjust this number via environment variable `OMP_NUM_THREADS` or at any time by calling `omp_set_num_threads(10)`. This function is available in Python through faiss.

For the `add` and `search` functions, threading is over the vectors. This means that querying or adding a single vector is not multi-threaded. 

GPU Faiss for a single GPU is not internally multi-CPU threaded.

## Performance of search

The best performance in terms of QPS is obtained when queries are submitted by batches.

If queries are submitted one by one, then they are executed in the calling thread (this is currently the case for all indexes). So it is also relatively efficient to have multiple threads call `search` for singleton queries. 
Python interface releases the Global Interpreter Lock for all calls, so using python multithreading will effectively use several cores. 

However it is very inefficient to call batches of queries from multiple threads, this will spawn more threads than CPU cores. 
Note that merely calling a library linked with OpenMP (like Faiss) incurs a runtime overhead when starting a new thread with pthread_create. This overhead is visible when many short-lived threads are spawned.
If this is an issue, compile Faiss without openmp (remove -openmp from the compilation options) and link with a single-threaded BLAS implementation (for MKL select the sequential version in https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html).

## Performance of internal threading (OpenMP)

Choosing the number of OpenMP threads is not always obvious. There are architectures where setting **fewer** threads than cores results in significantly **more** efficient execution. For example, on an Intel E5-2680 v2, it is useful to set the number of threads to 20 rather than the default 40.

When using the OpenBLAS BLAS implementation, it is useful to set the thread policy to PASSIVE with 

```
export OMP_WAIT_POLICY=PASSIVE 
```

see the discussion on issue [#53](https://github.com/facebookresearch/faiss/issues/53)

## Reproducibility with multiple threads

The general design of Faiss is that by default random seeds are fixed so that computations are deterministic. 
When computations are multi-threaded, this is true as well, since most computations can be balanced statically (`omp schedule static`).
Therefore, several runs of the same operation with the same number of threads give the same result bitwise. 
The known exceptions are: 

- functionalities that require LAPACK eigenvalue extraction are not reproducible exactly because the `esyev` function is not reproducible exactly (with MKL). This function is used in the training of `PCAMatrix` and `OPQMatrix`. The difference between different runs is in the order of the machine precision. Note that MKL does not by default guarantee that other operations, include matrix multiplication, are bit-exact reproducible, see this doc on [Conditional Numerical Reproducibility](https://software.intel.com/content/www/us/en/develop/articles/introduction-to-the-conditional-numerical-reproducibility-cnr.html).

- the `HNSW` add function is performed in an unspecified order. Implementing it with a static thread scheduling is too inefficient. Search is deterministic.

For these two cases, several runs will give different results. 

## Asynchronous search 

It can be useful to perform an `Index` search operation in parallel with some other computation including:

- single thread computations

- waiting for I/O

- GPU computations 

This way, the program runs in parallel. For Faiss CPU, it is **not useful** to parallelize with other multithreaded computations (eg. other searches), because this will spawn too many threads and degrade overall performance; multiple incoming searches from potentially different user threads should be enqueued and aggregated/batched by the user before handing to Faiss. 

It is of course possible and useful to run operations in parallel on multiple GPUs, where each CPU thread is dedicated to kernel launches on a different GPU, this is how `IndexProxy` and `IndexShards` are implemented.

How to spawn the search thread:

- in C++: with eg. `pthread_create` + `pthread_join`

- in Python: with eg. `thread.start_new_thread` + a lock, or with `multiprocessing.pool.ThreadPool`. The search, add and train functions release the Global Interpreter Lock.

<!--
- in Lua: it is bit more complicated because if the index was allocated in the main thread, Lua will try to serialize it, which does not work. Therefore, there is an `AsyncIndex` object that spawns a C++ thread that does the search. See the example [[https://phabricator.fb.com/diffusion/FBS/browse/master/fbcode/deeplearning/projects/faiss/tests/test_async.lua | tests/test_async.lua]].
-->
