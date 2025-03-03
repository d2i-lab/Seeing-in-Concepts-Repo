# Reporting issues

## Things to report 

We receive bug reports that are not actionable because impossible to reproduce. If you start a bug report, please make sure to include (use a [gist](https://gist.github.com/) if it is too bulky): 

- minimal code to reproduce the issue

- a gdb stacktrace if C++ crash

- please open a new issue unless you are certain that you have the exact same one as an existing one. This makes it easier for us to triage.

- if GPU: 

  - version of OS, type of GPU, nvidia-smi output if GPU 

  - output of ldd on executable or `_swigfaiss_gpu.so` for the Python version

Also, please consider the following notes on [how to productively report bugs](http://math-atlas.sourceforge.net/faq.html#utone).

Do not send e-mails for Faiss support (or call us on the phone (!)): we will redirect you to the issues page of Faiss.
Do not send screen shots of text, copy-paste text instead.

## Our issue workflow

We monitor issues actively. 
We label issues as: 
- *unconfirmed bug* = if what you report is correct, then it's a bug
- *cant-repro*  = we cannot repro because there is insufficient info or the bug does not appear when we test
- *bug*         = we verified there is a bug
- *feature request* = feature request acknowledged
- *enhancement* = we will consider implementing a fix (not necessarily soon)
- *documentation* = issue with the documentation (docstrings / C++ code comments / wiki)
- *install*     = install-time problem (compile, conda, etc.)
- *GPU* = GPU-specific issue
- *Performance* = performance related issue (parallelism, memory management, disk storage, performance regressions, etc.)
- *Implementation* = issue related to algorithm implementations or numerical issues
- *Integration* = issues related to how faiss interacts with other libraries/software
- *Platform* = Issues related to hardware that is neither x86 CPU nor GPU-specific; or non-Linux OS
- *ez* = Issues that a beginner contributor could support

We expect users to close the issues they open. 
However, we aggressively close issues after 15 days of inactivity, even if they are not fixed. 
Feel free to re-open if you have the exact same problem.

Previous tags around *questions*/*help wanted* will be moved to the [Discussions tab](https://github.com/facebookresearch/faiss/discussions).

## Our pull request workflow

We welcome contributions to Faiss, see [CONTRIBUTING](https://github.com/facebookresearch/faiss/blob/main/CONTRIBUTING.md). 
There is quite some overhead for us to accept PRs and include them in Faiss. 
Therefore, don't be surprised if it takes some time to accept.


# Compile errors

TODO  with realistic errors for the current build system

# Runtime errors

### Missing module

**ImportError: No module named swigfaiss**

This means that Faiss was not compiled. 


### GPU Precomputed table error

If you get the following assertion error when constructing an `GpuIndexIVFPQ` with precomputed tables:

```
Faiss assertion usePrecomputed_ || IVFPQ::isSupportedNoPrecomputedSubDimSize( this->d / subQuantizers_) failed in void faiss::gpu::GpuIndexIVFPQ::assertSettings_() const at GpuIndexIVFPQ.cu:469Aborted (core dumped)
```

Then make sure that the ratio between the dimension and the PQ size (d/M) is one of the values mentioned in this function:
 
https://rawgit.com/facebookresearch/faiss/master/docs/html/PQScanMultiPassNoPrecomputed_8cu_source.html#l00029

If this is not the case, you can pre-process the input vectors with an OPQ transform. 


### Slow brute-force search with OpenBLAS

The search performance with OpenBLAS can degrade seriously because of a bad interaction between OpenMP and OpenBLAS threads. To avoid this, `export OMP_WAIT_POLICY=PASSIVE` before running the executable. See issue [#53](https://github.com/facebookresearch/faiss/issues/53)

### Surprising FAISS, OpenMP and OpenBLAS interaction
When using FAISS with OpenBLAS on Ubuntu, please make sure you are using `libopenblas-openmp-dev`/`libopenblas0-openmp` instead of `libopenblas-dev` / `libopenblas0-pthread`. Otherwise, you may spawn an unintentional `N^2` threads: each of OpenMP `N` threads calls `sgemm()`, and each `sgemm()` instantiates its own `N` OpenBLAS threads. This is because OpenBLAS uses [pthread scheduling by default](https://github.com/OpenMathLib/OpenBLAS/wiki/Faq#multi-threaded), and it may trigger a weird interaction if code spawning OpenMP thread is run not from the main thread.

### Crashes in pure Python code 

This has been fixed in version 1.4.0, that introduced automatic tracking of c++ references in Python. 
If you see this error in a version >=1.4.0, please report it as an issue. 
 
### No GPU Faiss

The most common symptom is:
```
AttributeError: module 'faiss' has no attribute 'StandardGpuResources'
```
because `StandardGpuResources` is the GPU-specific object that is most often accessed first.

The current behavior of Faiss is to try to load the GPU Faiss and fallback to CPU if that fails. 
This is irrespective of whether Faiss was compiled with GPU support or not. 
No error is reported in case of failure. 

To actually see the error message, use
```
from faiss import _swigfaiss_gpu
```
which will output a more interpretable message like
```
ImportError: libcudart.so.9.2: cannot open shared object file: No such file or directory
```

### Slow GPU intialization

The first time you access a Faiss GPU function may be very slow if the Cuda code was not compiled specifically for your GPU generation. 
This is due to the CUDA caching system that precompiles kernels and stores them to `~/.nv`. 
If that directory is not writable or a dangling symlink or a very slow disk, then the slowness will happen every time Faiss is started, so it is useuful to make sure the directory is accessible.

## Known GPU issues

- For GPU faiss, `add` and `search` API calls need to be restructured somewhat to handle massive inputs in some cases, due to 32/64 bit integer confusion in various places. 32 bit integer math is much faster on the GPU, and this fact sadly leaked to the CPU side of GPU faiss. This is on the TODO list. Ideally, GPU faiss will handle any paging needed (so you can, say, pass a pointer to a 1 TB region of memory-mapped vectors to `add` or `search` and it would just work), but this requires some cleanup.

- Excessive memory requests on the GPU do not produce friendly errors (e.g., attempting to enable precomputed codes on massive databases with a large number of coarse centroids, which may require 5+ GB of free storage). We will try to intercept this and make it friendlier in the future.

## Deadlocking FAISS

It is possible to deadlock FAISS in Python / pybind11 interaction in the following scenario. Consider a simple pybind11 call from Python:
```python
import faiss

from faisstest_module import RunFaissTest

if __name__ == "__main__":
    RunFaissTest()
```

And the following C++ pybind11 module:
```c++
#include "faiss/Index.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/impl/io.h"
#include "faiss/index_factory.h"
#include "faiss/index_io.h"

void DoTest() {
  constexpr size_t numEmbeddings = 16384;
  constexpr size_t embeddingDim = 16;

  std::vector<float> input(numEmbeddings * embeddingDim);

  std::unique_ptr<faiss::Index> index(faiss::index_factory(embeddingDim, "PQ4np"));

  index->train(numEmbeddings, input.data());

  std::cout << "Trained!" << std::endl;
}

PYBIND11_MODULE(faisstest_module, m) {
  m.def("RunFaissTest", []() {
    std::thread thr([] {
      // PyThreadState* state = PyEval_SaveThread();
      DoTest();
      // PyEval_RestoreThread(state);
    });
    thr.join();
  });
}
```

The python code will deadlock unless the commented rows are uncommented, such as
```c++
      PyThreadState* state = PyEval_SaveThread();
      DoTest();
      PyEval_RestoreThread(state);
```

The reason is that the Python wrapper for FAISS installs a handler that checks whether Python wants to interrupt some computations. The handler interacts with GIL API. The handler is here https://github.com/facebookresearch/faiss/blob/main/faiss/python/swigfaiss.swig#L865 .

## Faiss function called with invalid data type

Some numpy functions return an `np.int64` instead of an int, eg. when accessing an `np.int64` array. 
Faiss functions wrapped in SWIG cannot consume this, so the number should be cast explicitly, eg. with 

```python
k = np.array([1, 3])[0]
D, I = index.search(x, k)       # error!
D, I = index.search(x, int(k))  # works
```