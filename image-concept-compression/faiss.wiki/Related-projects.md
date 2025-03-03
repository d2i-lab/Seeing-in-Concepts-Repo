This page references other projects and repositories that use Faiss or that are related to it. 
They are unaffiliated the main Faiss, so any issue with those should be redirected to the respective packages.
If you did something interesting with Faiss and you want to add a link to your project, just ask us (via an issue).

# Bindings to other languages and porting to other platforms

There are projects to interface Faiss with other languages, see 

- the Rust bindings at https://github.com/Enet4/faiss-rs

- the .NET Core 3.0 bindings at https://gitlab.com/josetruyol/faisssharp

- the Java interface at https://github.com/gameofdimension/jni-faiss

- ruby wrapper here https://github.com/ankane/faiss

- golang wrapper here:  https://github.com/DataIntelligenceCrew/go-faiss/

Other platforms: 

- Faiss is reported to work on Raspberry Pi (see https://github.com/facebookresearch/faiss/issues/1071) **update** Faiss runs on ARM server machines with Linux.

- Faiss compiles on PowerPC, see issues [#529](https://github.com/facebookresearch/faiss/issues/529) and [#1098](https://github.com/facebookresearch/faiss/issues/1098)

- Faiss compiles on the Apple M1, see [#2111](https://github.com/facebookresearch/faiss/issues/2111). We will soon add this as a supported platform.

Other packages: 

- Faiss C++ on CPU is available via a Brew package: https://github.com/Homebrew/homebrew-core/pull/49235

- Faiss is reported to work on Google collab: https://stackoverflow.com/questions/47967252/installing-faiss-on-google-colaboratory

- Faiss on Alibaba cloud: https://www.alibabacloud.com/help/en/e-mapreduce/latest/faiss-server

# Packages that use or extend Faiss

- QuickerADC, an implementation of IVFPQ variants using SIMD with better precision/speed operating points: https://github.com/technicolor-research/faiss-quickeradc See also the paper [Quicker ADC : Unlocking the hidden potential of Product Quantization with SIMD](https://arxiv.org/abs/1812.09162), André et al.

- pgANN: Approximate Nearest Neighbor (ANN) searches with a PostgreSQL database.  https://github.com/netrasys/pgANN

- milvus offers a uniform interface to Faiss + other packages and manages it like a database engine (client/server and the like): https://github.com/milvus-io/milvus

- vearch: a search engine based on Faiss with good scalability and real-time updates: https://github.com/vearch/vearch

- port of (an old version of) Faiss to Windows: https://github.com/bitsun/faiss-windows **update**: Faiss is now supported on windows.

- EFANNA, a k-NN graph construction method that can use Faiss: https://github.com/ZJULearning/efanna_graph

- Autofaiss, a library from Criteo that automatically selects a Faiss index: https://github.com/criteo/autofaiss

- Managed Faiss, a service built on top of Faiss with higher level functionalities: https://www.pinecone.io/managed-faiss

- Haystack retriever, that manages document stores: https://haystack.deepset.ai/

- OpenSearch k-NN a nearest-neighbor search service built on top of Faiss (and used in AWS) https://opensearch.org/docs/latest/search-plugins/knn/index/

# Alternative packages

There are many approximate nearest neighbor search libraries. 
You may be interested in:

- NMSLib (An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces): https://github.com/nmslib/nmslib

- an implementation of IVF-HNSW https://github.com/dbaranchuk/ivf-hnsw

- Annoy (Approximate Nearest Neighbors in C++/Python): https://github.com/spotify/annoy

- A promising GPU index here (vaporware as of 2020-01-20): https://github.com/cgtuebingen/ggnn 

- Google's SCANN package (very fast!): https://github.com/google-research/google-research/tree/master/scann

- Poeem (by JD.com): joint training of recommendation embeddings and index: https://github.com/jdcomsearch/poeem

- Pinecone: a fully fledged similarity search service (closed source, not based on Faiss) https://www.pinecone.io/

- Weaviate: a text indexing service with a custom HNSW implementation https://www.semi.technology/developers/weaviate/current/vector-index-plugins/

- Pyserini: a document indexing library that uses Lucene and Faiss https://github.com/castorini/pyserini

- Qdrant: a vector search engine based on HNSW that supports filtering https://qdrant.tech/

- Redis: a vector search engine that supports HNSW (k-nn and range search), https://redis.io/docs/stack/search/reference/vectors/

- Usearch: a templatized HSNW implementation with many language bindings, https://github.com/unum-cloud/usearch

- Epsilla: a vector search engine for low-dimensional vectors (?): https://github.com/epsilla-cloud/vectordb

- Spotify's Voyager: a HNSW implementation with speed and memory usage improvements: https://spotify.github.io/voyager/

- Puck is Baidu's search library, supports HNSW and PQ compression: https://github.com/baidu/puck

# Benchmarks

- Approximate nearest neighbor benchmark page: https://github.com/erikbern/ann-benchmarks (The benchmark is notoriously outdated even w.r.t. accepted PRs)

- Another benchmark: https://github.com/matsui528/annbench

- The billion-scale approximate nearest neighbor challenge: https://big-ann-benchmarks.com/ (Faiss was used as a baseline and is integrated into the winner solution of Track 1)

# To read & watch about Faiss

- Faiss tips by @matsui528: https://github.com/matsui528/faiss_tips

- Faiss presentation at the Milvus summit (nov 2020): https://www.youtube.com/watch?v=Un1Q92lfhPM

- a less terse tutorial than Faiss' official one: https://www.pinecone.io/learn/faiss-tutorial/ (James Briggs expanded it to a full-fledged
 book "Faiss: the missing manual")

- a series of video tutorials from James Briggs on FAISS and other similarity search topics: https://www.youtube.com/watch?v=B7wmo_NImgM&list=PLIUOU7oqGTLhlWpTz4NnuT3FekouIVlqc&index=4

- not really about Faiss, but about vector search engines like Weaviate: https://www.youtube.com/watch?v=WijYx9_Bpkw (this episode is with Yury Malkov, the inventor of HNSW). Episode with Matthijs Douze about quantization and Faiss: https://open.spotify.com/episode/7i1ytL29K2IRbeQJjXsEKZ

