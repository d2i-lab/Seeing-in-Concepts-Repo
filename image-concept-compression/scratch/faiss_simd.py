import os
import time
import argparse
import pandas as pd
from importlib import reload

import faiss
import numpy as np
from faiss.contrib.datasets import SyntheticDataset

def bench_time(index, database, times=1_000):
    t0 = time.time()
    for _ in range(times):
        search_query = database[np.random.choice(database.shape[0], 100)]
        # print(search_query.shape)
        index.search(search_query, 5)
    return time.time() - t0

def get_query_invlists_info(index, query):
    """
    Get information about which inverted lists a query falls into and their sizes.
    
    Args:
    - index: A trained FAISS IVF-PQ index
    - query: A numpy array of shape (d,) where d is the dimension of the index
    
    Returns:
    - A list of tuples, each containing (invlist_id, invlist_size)
    """
    if not isinstance(index, faiss.IndexIVFPQ):
        raise ValueError("The provided index must be an instance of faiss.IndexIVFPQ")
    
    # Ensure the query is a 2D array
    if query.ndim == 1:
        query = query.reshape(1, -1)
    

    _, assigned_ids = index.quantizer.search(query, index.nprobe)

    actual_results = []
    for batch in range(0, len(assigned_ids)):
        for invlist_id in assigned_ids[batch]:
            invlist = index.invlists.list_size(int(invlist_id))
            actual_results.append((int(invlist_id), invlist))

    # return result
    return actual_results


def run_benchmark(index, dataset, num_threads, nprobe_values, k=8):
    results = []

    for nprobe in nprobe_values:
        index.nprobe = nprobe
        
        total_points = 0
        search_time = 0
        
        queries = dataset.get_queries()
        
        t0 = time.time()
        for query in queries:
            
            _, _ = index.search(query.reshape(1, -1), k)
        
        search_time = time.time() - t0

        for query in queries:
            invlists_info = get_query_invlists_info(index, query)
            total_points += sum(size for _, size in invlists_info)
        
        
        results.append({
            'OMP_NUM_THREADS': num_threads,
            'nprobe': nprobe,
            'total_points': total_points,
            'runtime': search_time,
            'queries_per_second': total_points/ search_time
        })
    
    return results

def main():
    ds = SyntheticDataset(512, 10_000, 200_000, 5000)
    
    index = faiss.IndexFlatL2(ds.d)
    m = 32
    n_centroids = 1024
    nbits = 10
    ivfpq = faiss.IndexIVFPQ(index, ds.d, n_centroids, m, nbits)
    ivfpq.train(ds.get_train())
    ivfpq.add(ds.get_database())
    
    thread_values = [1, 2, 4, 8, 16, 32,]
    # nprobe_values = [1, 4, 16, 64, 256]
    nprobe_values = [1, 2, 4, 8, 16]

    
    all_results = []
    
    for num_threads in thread_values:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        faiss.omp_set_num_threads(num_threads)
        
        results = run_benchmark(ivfpq, ds, num_threads, nprobe_values)
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))
    
    # Optionally, save the results to a CSV file
    df.to_csv('benchmark_results.csv', index=False)

if __name__ == "__main__":
    main()
