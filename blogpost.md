# Introduction
## Graph Neural Networks

Many neural networks that operate on graphs work within the ‘message passing’ paradigm, where each layer of the network is responsible for aggregating ‘messages’ - functions of the node features - that are passed from a node to its neighbours. Adding depth to the network allows information from more distant nodes to be combined, as each subsequent layer allows information to be passed one edge further than the previous one.

This approach is a powerful one: by designing the network to process only local neighbourhoods, we allow weight sharing between all neighborhoods and allow the networks to process graphs with arbitrary sizes and topologies.

 However, this focus on local information can make it difficult to apply the GNN framework when interactions between distant nodes are important. We describe such datasets as exhibiting ‘long range interaction’ (LRI).  Recent work has shown the message passing paradigm can fail in some surprising ways on LRI problems.

First, [REF] identified ‘over-smoothing’, where adding too many layers to a GNN can cause nearby nodes to have indistinguishable hidden features in the later layers of the network. This occurs because each convolution blurs together the features within a neighbourhood [ TODO more?]. This is especially an issue in the LRI  case, because a large number of layers is required for messages to reach between nodes that are far apart.

Second, [REF] identified ‘over-squashing’, where the graph topology induces bottlenecks that prevent the flow of information between different parts of the graph. Because each message in an MPNN has a fixed capacity, nodes with many neighbours may not be able to pass on all the useful information that they have access to. LRI tasks should therefore be harder to solve in topologies that have strict bottlenecks, because essential information is more likely to be lost while passing from node to node.


## The Long Range Graph Benchmark

Many of the papers that propose methods in the LRI space have tested their approach on toy datasets - while this is useful, it can give an unrealistic depiction of the weaknesses of new approaches. Furthermore, many existing benchmark graph datasets are best solved by shallow MPNNs that only consider local information, and so will not benefit from even the most well-founded LRI methods (REF). 

The Long Range Graph Benchmark (REF) are a number of datasets that attempt to provide a common framework for testing and benchmarking new LRI methods. Putatively, these are real world datasets with tasks that can only be solved by successfully solving the LRI problem, and so provide an effective test of any new LRI method. There are five datasets in total - we will describe each of these briefly below.

PascalVOC-SP: this dataset was derived from the Pascal 2011 image dataset (REF), which has class labels associated to every pixel. Each image was segmented into superpixels, and the task is to predict the class of the pixel that was originally at the centroid of each pixel. A graph is formed where the nodes correspond to each superpixel, the node features are statistics RGB values within each pixel, and the edges correspond to which superpixels are contiguous in the image. 
COCO-SP: this is similar to PascalVOC-SP, but was derived from the MS COCO dataset (REF).
PCQM-Contact: this dataset was derived from the PCQM4M (REF) molecular dataset, where each node is an atom, and the edges correspond to the molecular structure. The task is to predict pairs of nodes that will be less than 3.5 angstroms apart in the final configuration of the molecule. To ensure that only ‘long-range’ predictions are counted, the task is limited to pairs of molecules that are separated by at least 5 hops.
Peptides-func and peptides-struct: these are derived from the SATPdb (REF) dataset of peptides, a class of molecules that is characterised by a large number of nodes and complex structure. While typical peptide datasets use nodes to represent amino acids, the authors instead split these into multiple nodes, each representing individual atoms. In doing so, they imposed extra separation between the graphs. Then they defined two tasks, one a graph-level regression, one a graph-level classification, to predict molecular properties of the graph.


## Are these truly ‘long range’ benchmarks?

The central claim of the paper is that the above datasets provide a benchmark for assessing whether a new method is capable of testing new LRI methods. In this section, we describe the arguments that the authors make in support of this claim, and discuss their strengths and weaknesses.
 
In the case of one dataset -  PCQM-Contact - the task was explicitly constructed to require long range interactions. In our view, this is an effective way of ensuring that the task is LRI dependent, but we note that it doesn’t apply for 4 out of 5 datasets.
The authors showed that transformer architectures, which ignore the input graph in favour of a fully connected one, outperformed all other methods in 4 out of the 5 datasets. While they interpreted this as evidence of LRI in the data, there are other plausible explanations. For example, it’s possible that the extra expressivity afforded by the transformers attention mechanism was responsible for the improved performance. Arguably, we should look for more direct evidence that the transformers were capable of leveraging long range interactions to improve performance.
They generated statistics characterising the graphs found in each dataset, such as the graph diameter, the average shortest path, and the number of nodes. They claimed that high values of these statistics indicated LRI within the graph. However, it’s not clear whether these statistics actually capture features of the graph topology relevant to the LRI problem.


# Motivation
( weaknesses/strengths/potential which triggered your group to come up with a response.)


# Contribution
(some intro line about motivation....)

We summarise our main contributions as follows:
1. We apply curvature-based rewiring method -  Stochastic Discrete Ricci Flow (SDRF) algorithm to Pascal SP [[2]](#2). As shown in [Figure 2](#fig2), we rewire the graph and pass it through trained graph models. The aim is to analyse whether rewiring helps to mitigate the problem of oversquashing.  
2. Pascal-SP dataset has node embedding of size 14 where the first 12 features are related to the color of the node and the last 2 features are related to its x,y position. We separate the feature embedding into node embedding and coordinate embedding, to incorporate type-0 representations (i.e. relative distances between the nodes). We do this, inorder to see whether considering geometrical representations of the nodes can help with oversquashing. We implement E(n)-invariant and E(n)-equivariant graph networks [[3]](#3) to see whether these 2 variants of translation equivariance architectures help to handle oversquashing. 
3. To handle the issue of oversmoothing where the values of hidden node converge in deeper layers, we apply some of the "jumping techniques" introduced in [[4]](#4).  Specifically, we apply concatenation and max pooling of all the layers in every forward pass. 
4. We measure the sensitivity of node x to node y, or the influence of y on x, by measuring how much a change in the input feature of y affects the representation of x in the last layer. This helps us to see how for different models, the influences over short and long graph distances change. We implement the definition of this influence score introduced in Section 3, Definition 3.1 of [[4]](#4).
5. We analyse the usage of the Cheeger constant [[2]](#2) and minimal average path as "measurements of LRI" in the graph by attempting to find a correlation between the values of these metrics and the level of influence of long-range interactions on different models' prediction.
We predict that high Cheeger values would correlate with high bottleneck and high average shortest path would correlate with the range of interactions. As the oversquashing problem relates to an interaction between bottlenecking and distance, we predict that models that should perform better under LRI tasks would specifically perform better on
graphs with high values on both of these metrics.
6. Finally advanced  models like Steerable GNN [[5]](#5) have been used mostly in toy datasets like N-body and QM9. We experiment this with Pascal-SP to verify whether steerable messages improve upon the above mentioned trivial equivariant graph networks that send invariant messages. Ideally we expect this message passing approach, should be maximally expressive due to E(3) equivariance. (This is an ongoing Work)



| <img width="1246" alt="image" src="https://github.com/madhurapawaruva/uva-dl2-team11-forpeer/assets/117770386/9b0c9463-008f-47b7-817a-9a63c796e8a7">    | <img width="739" alt="image" src="https://github.com/madhurapawaruva/uva-dl2-team11-forpeer/assets/117770386/ed650fa6-ec70-4c9f-9594-87bcddc3cff2" id="fig2"> | 
| -------- | -------- |
| Figure 1: E(n)-Invariant and E(n)-Equivariant Architecture    | Figure 2: Rewiring Inference Architecture  |  
  
  | <img src="assets/cheeger-median-against-path.png"> | <img src="assets/heat-map-three-values.png"> | <img src="assets/histogram-graph-values-destribution.png"> |
 | -------------------------------------------|-------------------------------|-----------------------------------------------------------|
 | Figure 3: Cheeger value against average shortest path | Figure 4: heat map of graph diameter against Cheeger value and average shortest path | Figure 5: Distribution of graphs across the 3 metrics |
# Results

Applying SDRF rewiring to the graphs of Pascal dataset and then training the Transformer+LapPE model gives an improved performance as shown in the table below. We are working on applying the same for other models.

| No. of edge additions | Best test F1 |
| ------------------- | ------------ |
| 0 (original graphs) | 0.261        |
| 10                  | 0.2757       |
| 20                  | 0.2635       |

We see an increase in f1 scores on adding 10 edges to every graph, but we also see a decrease in score on adding 20 edges. It would be interesting to experiment with the amount of edges being added and the effect it has on f1 scores to reach an appropriate threshold after which rewiring becomes detrimental (ongoing work).

In the table below, we present the F1 scores for the models we trained. Here JK1 denotes the jumping knowledge variant 1 where we concatenate hidden outputs of all layers. And JK2 denotes the jumping knowledge variant where we do maximum pooling of all the layers.

| Model                   | # Params  | Best train F1  | Best val F1 | Best test F1 |
|-------------------------|-----------|----------------|-------------|--------------|
| GCN                     | 496k      | 0.46046        | 0.15339     | 0.1585       |
| E(n)-Invariant          | 523k      | 0.44664        | 0.21416     | 0.2213       |
| E(n)-Invariant (JK 1)   | 572k      | 0.38194        | 0.22385     | 0.23597      |
| E(n)-Invariant (JK 2)   | 523k      | 0.51587        | 0.23583     | 0.23675      |
| E(n)-Equivariant        | 523k      | 0.3767         | 0.2434      | 0.2516       |
| E(n)-Equivariant (JK 1) |           |                |             |              |
| E(n)-Equivariant (JK 2) | 523k      | 0.4613         | 0.2399      | 0.2453       |
| Transformer+LapPE       | 501k      | 0.8062         | 0.2624      | 0.2610       |

(influence score dist)

We use [Figure 3](#fig3) to select a few graphs from the datasets
to plot and observe whether they suffer from long range interactions and bottlenecks.
We would generate a heat map similar to [Figure 4](#fig4) with accuracies instead of diameter per model
in order to observe the relation between the metrics and the model behaviors.
# Conclusion

# References
<a id="1">[1]</a> 
Dwivedi, Vijay Prakash et al. “Long Range Graph Benchmark.” ArXiv abs/2206.08164 (2022): n. pag.

<a id="2">[2]</a> 
Topping, Jake et al. “Understanding over-squashing and bottlenecks on graphs via curvature.” ArXiv abs/2111.14522 (2021): n. pag.

<a id="3">[3]</a>
Satorras, Victor Garcia et al. “E(n) Equivariant Graph Neural Networks.” International Conference on Machine Learning (2021).

<a id="4">[4]</a>
Xu, Keyulu et al. “Representation Learning on Graphs with Jumping Knowledge Networks.” International Conference on Machine Learning (2018).

<a id="5">[5]</a>
Brandstetter et al. "Geometric And Physical Quantities Improve E(3) Equivariant Message Passing"



