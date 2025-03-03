Locality Sensitive Hashing (LSH) is an indexing method whose theoretical aspects have been studied extensively. For most application cases it performs worse than PQ in the tradeoffs between memory vs. accuracy and/or speed vs. accuracy. There has is renewed interest in LSH variants following the publication of the bio-inspired "Fly indexing" algorithm []. 

However, most LSH papers avoid directly comparing with PQ. This is a bit surprising, since one of the main datasets that is used for comparison is SIFT1M, that was introduced simultaneously with PQ in the same paper. 

In the following, we provide points of comparison with a few other papers, and with Faiss' own implementation of LSH, and short code snippets that show these results. The datasets we experiment with are SIFT1M (ref) and Glove (ref). They can be loaded with:

```python



```
## References

### The fly algorithm

```
@article{dasgupta2017neural,
  title={A neural algorithm for a fundamental computing problem},
  author={Dasgupta, Sanjoy and Stevens, Charles F and Navlakha, Saket},
  journal={Science},
  volume={358},
  number={6364},
  pages={793--796},
  year={2017},
  publisher={American Association for the Advancement of Science}
}
```
http://science.sciencemag.org/content/358/6364/793 
https://www.biorxiv.org/content/biorxiv/early/2017/08/25/180471.full.pdf

Comparison on SIFT, Glove, MNIST

### Modern version of LSH
```
@inproceedings{heo2012spherical,
  title={Spherical hashing},
  author={Heo, Jae-Pil and Lee, Youngwoon and He, Junfeng and Chang, Shih-Fu and Yoon, Sung-Eui},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on},
  pages={2957--2964},
  year={2012},
  organization={IEEE}
}
```

https://sglab.kaist.ac.kr/Spherical_Hashing/

Provides code 

https://sglab.kaist.ac.kr/Spherical_Hashing/SphericalHashing_TPAMI.pdf

Comparison on GIST in 960D and 384D




## Comparison with classical LSH

Note about symmetric vs. asymmetric PQ 


## Comparison with the "Fly algorithm"

## Comparison with ITQ 

## A note about supervised indexing methods

There are a few works that attempt to use supervision to improve similarity search. This setup was analyzed in [How should we evaluate supervised hashing? Sablayrolles & al.](https://arxiv.org/abs/1609.06753). The experimental protocol consists in defining "similar" vectors as vectors from the same class. This protocol transforms the similarity search problem into a classification problem: the optimal way of performing the search is to classify the query and database vectors.



Another useful resource is [Ann-Benchmarks](https://github.com/erikbern/ann-benchmarks), although we of course dispute their result that Faiss is not the fastest package out there! 