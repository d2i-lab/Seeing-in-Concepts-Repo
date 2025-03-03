This page introduces two low-level (ie. only C++) APIs to generalize the inverted list storage. 
This is useful eg. to store lists in a key-value database. 

The `InvertedLists` abstract class defines how inverted lists are accessed from the search code. 
Any object that offers this interface can be used to store the lists. 

The `InvertedListsScanner` offers even finer-grained access because the scanning function can be called on user-provided id and code tables. 


# Recap about `IndexIVF`

The `IndexIVF` class (and its children) is used for all large-scale applications of Faiss. 
It clusters all input vectors into `nlist` groups (`nlist` is a field of `IndexIVF`). 
At add time, a vector is assigned to a groups. 
At search time, the most similar groups to the query vector are identified and scanned exhaustively.

Thus, the `IndexIVF` has two components: 

- the quantizer (aka coarse quantizer) index. 
Given a vector, the search function of the quantizer index returns the group the vector belongs to. 
When searched with nprobe>1 results, it returns the nprobe nearest groups to the query vector (`nprobe` is a field of `IndexIVF`).

- the `InvertedLists` object. 
This object maps a group id (in 0..nlist-1), to a sequence of (code, id) pairs. 

# The `InvertedLists` object

Codes are arbitrary byte strings of a constant size `code_size`. 
For example, an `IndexIVFFlat` in 36D has code_size = 36 * sizeof(float) = 144 bytes.
Ids are arbitrary 64-bit integers (but negative values are reserved, so that's 63 useful bits).

In reality the codes and ids are returned in two separate arrays because some applications may need only one of them, and the memory alignement requirements are not the same.

## Methods for search

The three relevant methods of the object are 
```cpp
/// get the size of a list
size_t list_size(size_t list_no);

/// @return codes    size list_size * code_size
const uint8_t * get_codes (size_t list_no);
void release_codes (const uint8_t *codes) const;

/// @return ids      size list_size
const idx_t * get_ids (size_t list_no);
void release_ids (const idx_t *ids) const;
```
Pointers obtained through `get_codes` (resp `get_ids`) should be released with `release_codes` (resp `release_ids`). The object `InvertedLists::ScopedCodes` (resp `InvertedLists::ScopedIds`) can be used if you prefer [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization).

There is an additional `prefetch_lists` method, that is used by the search method to inform the `InvertedLists` object that some inverted lists will be required in the near future. 

Thus, the calling sequence of search looks like:

```
search(v) {
    list_nos = quantizer->search(v, nprobe)
    invlists->prefetch(list_nos)
    foreach no in list_nos {
        codes = invlists->get_codes(no)
        // linear scan of codes
        // select most similar codes 
        ids = invlists->get_ids(no)
        // convert list indexes to ids
        invlists->release_ids(ids)
        invlists->release_codes(codes)
    }
}
```

## Methods for add()

Adding vectors requires read-write access to the InvertedLists object. 
This is provided by the `add_entries` method. 
Additional methods `update_entries` and `resize` are used for bulk operations like merging, splitting and removing elements.

# Behavior of the InvertedLists object

## Memory management

The `InvertedLists` object is deleted by the `IndexIVF` destructor if `own_invlists` is true. 
By default an `ArrayInvertedLists` object is constructed when the `IndexIVF` is instanciated, and `own_invlists` is set to true.
The default `invliststs` can be replaced with `replace_invlists`, and the user has to decide about ownership. 

## Threading

Read-only multithreaded access is allowed. 
There are some comments about concurrent read-write access in the code.

## I/O

The `InvertedLists` object needs not to be stored along with the index object. 
If this is not the case, the index object just contains the necessary information to handle external storage.

# Built-in `InvertedLists` classes

The `InvertedLists` class is designed with extensibility in mind. 
However, there are two built-in `InvertedLists` classes in Faiss.

## ArrayInvertedLists

This is basically a `std::vector<std::vector<...> >`. 
It is the simplest in-RAM inverted lists implementation, with very little overhead and fast add times.
It has a special status because it is instantiated automatically when an `IndexIVF` is built, so that vectors can be added right away.

## OnDiskInvertedLists

The inverted list data is stored on a memory-mapped file on disk.
There is an indirection table that maps list ids to an offset in the file. 
Since the data is memory-mapped, there is no need to explicitly fetch the data from disk.
However, the prefetch function is useful to exploit parallel reads on distributed file systems.

Interestingly, a "normal" `IndexIVF` can be loaded into an `OnDiskInvertedLists` by setting the `IO_FLAG_MMAP` flag to `read_index`.
This makes it possible to load an arbitrary number of indexes without worrying about whether they fit in RAM.

See the chapter about this topic [here](https://github.com/facebookresearch/faiss/wiki/Storing-IVF-indexes-on-disk)

## `HStackInvertedLists` and `VStackInvertedLists`

These are inverted list objects that combine several inverted list objects into one, so that they appear to be a single inverted lists object. 
The hstack/vstack name comes from the numpy functions of the same name. Indeed an inverted list object can be seen as a sparse matrix.
`HStack` combines a set of inv lists together so that the search is performed in the combination of all the inverted lists. 
`VStack` combines k InvertedLists objects of size nlist1 assuming into an inverted lists object of size nlist1 * k.


# The `InvertedListScanner` object

The inverted list scanning can be controlled outside of Faiss. 
This makes is unnecessary to implement the list access functions as callbacks, which is not always convenient. 

To support this, Faiss offers: 

- an `encode_vector` function that computes the inverted list codes into an array that can be used to populate inverted lists that are not managed by Faiss

- an `InvertedListScanner` object that can be obtained from an IndexIVF class. It can scan lists or compute a single query-to-code distance.

This access is at a very low level, but the user has total control over the scanning without having to implement callbacks as with the `InvertedLists` object.

## Encoding vectors

To encode the vectors, the calling code should: 

- quantize the vector to find the inverted list where it has to be stored

- call `encode_vectors` to actually encode it. 

Both functions operate on batches for efficiency.


Here is a simplified code that adds `nb` vectors stored in `xb` to custom inverted lists:
```c++
// size nb
idx_t *list_nos = ... ; 
// find inverted list numbers 
index_ivf.quantizer->assign (nb, xb, list_nos);

// size index->code_size * nb
uint8_t *codes = ...; 
// compute the codes to store
index->encode_vectors (nb, xb, list_nos, codes);

// populate the custom inverted lists 
for (idx_t i = 0; i < nb; i++) {
      idx_t list_no = list_nos[i]; 
      // allocate a new entry in the inverted list list_no
      // get a pointer to the new entry's id and code
      idx_t * id_ptr = ... 
      uint8_t * code_ptr =  ...
      // assume the vectors are numbered sequentially
      *id_ptr = i; 
      memcpy (code_ptr, codes + i * index->code_size, index->code_size);
}
```


See here for an example: [test_lowlevel_ivf.cpp (add)](https://github.com/facebookresearch/faiss/blob/master/tests/test_lowlevel_ivf.cpp#L115)

## Searching 

To perform a search, there are several loop levels. 

Here is a simplified code that performs the query. It queries `nq` vectors `xq` in `index`. 

```c++
// size nprobe * nq
float * q_dis =  ...
idx_t *q_lists = ...
// perform quantization manually
index.quantizer->search (nq, xq, nprobe, q_dis, q_lists); 

// get a scanner object
scanner = index.get_InvertedListScanner();

// allocate result arrays (size k * nq), properly initialized
idx_t *I = ...
float *D = ...

// loop over queries
for (idx_t i = 0; i < nq; i++) {
    // set the current query
    scanner->set_query (xq + i * d);

    // loop over inverted lists for this query
    for (int j = 0; j < nprobe; j++) {
        // set the current inverted list
        int list_no = q_lists[i * nprobe + j];
        scanner->set_list (list_no, q_dis[i * nprobe + j]);

        // get the inverted list from somewhere
        long list_size = ...
        idx_t *ids = ....
        uint8_t *codes = ....
        // perform the scan, updating result lists
        scanner->scan_codes (list_size, codes, ids, D + i * k, I + i * k, k); 
   }
   // re-order heap in decreasing order
   maxheap_reorder (k, D + i * k, I + i * k);
}
```

See details here:  [test_lowlevel_ivf.cpp (search)](https://github.com/facebookresearch/faiss/blob/master/tests/test_lowlevel_ivf.cpp#L152)


The scanner object is not thread-safe, but several of them can be used to process queries or inverted lists in parallel (see the same test for an example).