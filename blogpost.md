# Introduction
(some basic introduction)

(related work)

(analysis of the paper and its key components)
Message-passing graph neural networks (MP-GNNs) work by aggregating information locally within the neighborhoods of each node. This approach is powerful, but it suffers from a number of shortcomings in “long range interaction” (LRI) tasks, which require combining information from distant nodes. The paper [[1]](#1) introduces Long Range Graph Benchmark - with five graph datasets -  PascalVOC-SP, COCO-SP, PCQM-Contact, Peptides-func and Peptides-struct. PascalVOC-SP and COCO-SP are node classification datasets extracted from image datasets.  PCQM-Contact, Peptides-func and Peptides-struct are molecular datasets aimed at predicting properties of molecules or classifying them.   According to the paper, these datasets possess properties to model long range interactions by various graph learning tasks. These properties are huerestically defined in the paper, such as graph size, the nature of graph learning task and the contribution of global graph structure to the task. Along with introducing the datasets, the authors present baseline benchmarking results on the long range graph tasks using GNN and Graph Transformers. 

# Motivation
( weaknesses/strengths/potential which triggered your group to come up with a response.)


# Contribution

(talk about ricci flow)

(talk about enn )

(talk about egnn )

| <img width="1246" alt="image" src="https://github.com/madhurapawaruva/uva-dl2-team11-forpeer/assets/117770386/9b0c9463-008f-47b7-817a-9a63c796e8a7">    | <img width="739" alt="image" src="https://github.com/madhurapawaruva/uva-dl2-team11-forpeer/assets/117770386/ed650fa6-ec70-4c9f-9594-87bcddc3cff2">	| 
| -------- | -------- | 
| Figure 1: E(n)-Invariant and E(n)-Equivariant Architecture    | Figure 2: Rewiring Inference Architecture   | 
  
# Results

(talk about result of ricci flow)

(enn result)

(egnn result)

| Model                 | Best train F1 | Best val F1 | Best test F1 |
| --------------------- | ------------- | ----------- | ------------ |
| GCN                   |     0.46046		|    0.15339|   0.1585   |
| E(n)-Invariant |        0.44664	|  0.21416 | 			0.2213 |
| E(n)-Invariant (JK 1) |    0.38194	          |   	0.22385          |      0.23597         |
| E(n)-Invariant (JK 2)  |      0.51587	       |   0.23583          |       	0.23675         |
| E(n)-Equivariant  |         0.3767	     |    0.2434         |    	0.2516           |
| E(n)-Equivariant (JK 1) |              |           |    	          |
| E(n)-Equivariant (JK 2) |              |           |             |
| Transformer+LapPE     |     0.8062           |   0.2624           |   0.2610           |

(influence score dist)
# Conclusion

# References
<a id="1">[1]</a> 
Dwivedi, Vijay Prakash et al. “Long Range Graph Benchmark.” ArXiv abs/2206.08164 (2022): n. pag.

