Faiss can leverage your nvidia GPUs almost seamlessly.

First, declare a GPU resource, which encapsulates a chunk of the GPU memory:

### In Python

```python
res = faiss.StandardGpuResources()  # use a single GPU
```

### In C++

```C++
faiss::gpu::StandardGpuResources res;  // use a single GPU
```

Then build a GPU index using the GPU resource:

### In Python

```python
# build a flat (CPU) index
index_flat = faiss.IndexFlatL2(d)
# make it into a gpu index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
```

### In C++

```C++
faiss::gpu::GpuIndexFlatL2 gpu_index_flat(&res, d);
```

Note: a single GPU resource object can be used by multiple indices, 
as long as they are not issuing concurrent queries.

The obtained GPU index can be used the exact same way as a CPU index:

### In Python
```python
gpu_index_flat.add(xb)         # add vectors to the index
print(gpu_index_flat.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

### In C++

```C++
    gpu_index_flat.add(nb, xb);  // add vectors to the index
    printf("ntotal = %ld\n", gpu_index_flat.ntotal);

    int k = 4;
    {       // search xq
        idx_t *I = new idx_t[k * nq];
        float *D = new float[k * nq];

        gpu_index_flat.search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }
```

### Results

The results are the same as for the CPU version. Note also that the 
performance increase will not be noticeable on a small dataset.


## Using multiple GPUs

Making use of multiple GPUs is mainly a matter of declaring several
GPU resources. In python, this can be done implicitly using the 
`index_cpu_to_all_gpus` helper.

Examples:

### In Python

```python
ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)

gpu_index.add(xb)              # add vectors to the index
print(gpu_index.ntotal)

k = 4                          # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k) # actual search
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries
```

### In C++

```C++
    int ngpus = faiss::gpu::getNumDevices();

    printf("Number of GPUs: %d\n", ngpus);

    std::vector<faiss::gpu::GpuResources*> res;
    std::vector<int> devs;
    for(int i = 0; i < ngpus; i++) {
        res.push_back(new faiss::gpu::StandardGpuResources);
        devs.push_back(i);
    }

    faiss::IndexFlatL2 cpu_index(d);

    faiss::Index *gpu_index =
        faiss::gpu::index_cpu_to_gpu_multiple(
            res,
            devs,
            &cpu_index
        );

    printf("is_trained = %s\n", gpu_index->is_trained ? "true" : "false");
    gpu_index->add(nb, xb);  // vectors to the index
    printf("ntotal = %ld\n", gpu_index->ntotal);

    int k = 4;

    {       // search xq
        idx_t *I = new idx_t[k * nq];
        float *D = new float[k * nq];

        gpu_index->search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    delete gpu_index;

    for(int i = 0; i < ngpus; i++) {
        delete res[i];
    }
```