# Introduction
(some basic introduction)

(analysis of the paper and its key components)
Message-passing graph neural networks (MP-GNNs) work by aggregating information locally within the neighborhoods of each node. This approach is powerful, but it suffers from a number of shortcomings in “long range interaction” (LRI) tasks, which require combining information from distant nodes. The paper [[1]](#1) introduces Long Range Graph Benchmark - with five graph datasets -  PascalVOC-SP, COCO-SP, PCQM-Contact, Peptides-func and Peptides-struct. PascalVOC-SP and COCO-SP are node classification datasets extracted from image datasets.  PCQM-Contact, Peptides-func and Peptides-struct are molecular datasets aimed at predicting properties of molecules or classifying them.   According to the paper, these datasets possess properties to model long range interactions by various graph learning tasks. These properties are huerestically defined in the paper, such as graph size, the nature of graph learning task and the contribution of global graph structure to the task. Along with introducing the datasets, the authors present baseline benchmarking results on the long range graph tasks using GNN and Graph Transformers. 

# Motivation

# Contribution

# Results

# Conclusion

# References
<a id="1">[1]</a> 
Dwivedi, Vijay Prakash et al. “Long Range Graph Benchmark.” ArXiv abs/2206.08164 (2022): n. pag.

