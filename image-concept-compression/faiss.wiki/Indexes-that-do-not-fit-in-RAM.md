When an index does not fit in RAM, even after compression, there are several ways of handling it: 

- distribute ("shard") the index over several machines

- store the index on disk (possibly on a distributed file system)

- store the index in a distributed key-value store. 

In all cases, this incurs a runtime penalty compared to standard indexes stored in RAM of the local machine.

# General approach

The general approach is to: 

1. train the index (typically an IVF index), without adding data to it. Store the trained empty index.

2. split the database into parts. Load the trained index and add the parts independently. This can be done independently on several machines.

These two operations are straightforward. The subsequent operations depend on what storage exactly is used afterwards.

# Distributed index

Faiss comes with a simple RPC library to access indexes from several machines ("slaves"). 
Each slave contains an index with a part of the data (shard). 
At search time, the master issues the query to all slaves, which process it. Then the master combines the results into the output.

**Note: the server & RPC code provided with Faiss is for demonstration purposes only and does not include certain security protections. It is not meant to be run on an untrusted network or in a production environment.**

## client-server demo

Here is a demo on how to do this: [demo_client_server_ivf.py](https://github.com/facebookresearch/faiss/blob/master/demos/demo_client_server_ivf.py)

Start servers: 
```bash
for st in {5..8}; do python demo_client_server_ivf.py $st & done
```
Servers are run on local machine for the demo. 
For a real run, the servers should be launched on different machines with some cluster operating software and the machine + ports should be provided.

Perform search:
```
python demo_client_server_ivf.py 9
```


# On-disk storage

Faiss supports storing IVF indexes in a file on disk and accessing the file on-the-fly.

## Data layout

The content of the inverted lists is stored in a file that is distinct from the index file. 
By default the file layout is just raw data with 

```
[codes for inv list #0]
[ids for inv list #0]
[codes for inv list #1]
[ids for inv list #1]
...
[codes for inv list #nlist-1]
[ids for inv list #nlist-1]
```

The file is memory mapped at search time. 

## The `OnDiskInvertedLists` object

The `OnDiskInvertedLists` object contains contains an indirection table that maps an inverted list number to an offset in the file on disk. 
This table is kept in RAM so that accessing an inverted list requires a single disk seek.
When the index is written to a file, the offsets table is written in the index file.

## Loading a standard index on disk

A standard index can be memory-mapped as an on-disk index with using the I/O flag `IO_FLAG_MMAP`. 
This makes it possible to load many indexes that would not otherwise fit in RAM.

## Building an on-disk index

The on-disk index is built by merging the sharded indexes into one big index with `OnDisk.merge_from`

`OnDiskInvertedLists` does support adding vectors to the index, but it is very inefficient, and this support will likely be removed in some version of Faiss.

See [demo_ondisk_ivf.py](https://github.com/facebookresearch/faiss/blob/master/demos/demo_ondisk_ivf.py) for a demo on how to do this.

## Search performance

SSD has a much better performance than spinning disk for short reads with random accesses. 

Here is a small Python benchmark that reads blocks of arbitrary data from disk, with results: 

- on spinning disk: [spinning](https://gist.github.com/mdouze/1aa85afd3a753a6638106c9c06ed5f96)

- on SSD: [SSD](https://gist.github.com/mdouze/e40d0d70923d376a7d49a7ec45fe10b1)

### Caching 

Note that there is a subtle intermix between RAM and disk caching. 
The data may be stored on disk but due to the kernel buffering mechanism, the data is actually accessed from RAM. 
This may happen in two cases:

- when the dataset is small (like the demo above)

- when the same search is repeated several times. In that case, the inverted lists en up in the kernel cache buffers, so the search is fast after the first search.

### OS support

The on-disk functionality relies on "advanced" functionality from the host OS and compiler. 
It is known to work on native Linux and OSX. 
It is known to fail in at least the following cases: 

- when using address sanitizers like [ASAN](https://github.com/google/sanitizers/wiki/AddressSanitizer), the memory overhead is too high.

- some virtual machines do not implement memory mapping properly, see [issue #1262](https://github.com/facebookresearch/faiss/issues/1262)
