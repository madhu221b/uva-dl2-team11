# 1. Introduction
## 1.1 Graph Neural Networks

TODO need to reference this section. 

Many neural networks that operate on graphs work within the ‘message passing’ paradigm, where each layer of the network is responsible for aggregating ‘messages’ - functions of the node features - that are passed from a node to its neighbours. Adding depth to the network allows information from more distant nodes to be combined, as each subsequent layer allows information to be passed one edge further than the previous one. This approach is a powerful one: by designing the network to process only local neighbourhoods, we allow weight sharing between all neighborhoods and allow the networks to process graphs with arbitrary sizes and topologies. However, this focus on local information can make it difficult to apply the Message Passing framework when interactions between distant nodes are important. We describe such datasets as exhibiting ‘long range interaction’ (LRI).  Recent work has shown the message passing paradigm can fail in some surprising ways on LRI problems.

First, [[7]](#7) identified ‘over-smoothing’, where adding too many layers to a GNN can cause nearby nodes to have indistinguishable hidden features in the later layers of the network. This occurs because each convolution blurs together the features within a neighbourhood. This is especially an issue in the LRI  case, because a large number of layers is required for messages to reach between nodes that are far apart.

Second, [[12]](#2) identified ‘over-squashing’, where the graph topology induces bottlenecks that prevent the flow of information between different parts of the graph. Because each message in a Message Passing Neural Network (MP-NN) has a fixed capacity, nodes with many neighbours may not be able to pass on all the useful information that they have access to. LRI tasks should therefore be harder to solve in topologies that have strict bottlenecks, because essential information is more likely to be lost while passing from node to node.


## 1.2 The Long Range Graph Benchmark

Many of the papers that propose methods in the LRI space have tested their approach on toy datasets - while this is useful, it can give an unrealistic depiction of the weaknesses of new approaches. Furthermore, many existing benchmark graph datasets are best solved by shallow MPNNs that only consider local information, and so will not benefit from even the most well-founded LRI methods [[1]](#1). 

The Long Range Graph Benchmark [[1]](#1) are a number of datasets that attempt to provide a common framework for testing and benchmarking new LRI methods. Putatively, these are real world datasets with tasks that can only be solved by successfully solving the LRI problem, and so provide an effective test of any new LRI method. There are five datasets in total - we will describe each of these briefly below.

1. __PascalVOC-SP__: This dataset was derived from the Pascal 2011 image dataset [[8]](#8), which has class labels associated to every pixel. Each image was segmented into superpixels, and the task is to predict the class of the pixel that was originally at the centroid of each pixel. A graph is formed where the nodes correspond to each superpixel, the node features are statistics RGB values within each pixel, and the edges correspond to which superpixels are contiguous in the image. 
2. __COCO-SP__: this is similar to PascalVOC-SP, but was derived from the MS COCO dataset [[9]](#9).
3. __PCQM-Contact__: This dataset was derived from the PCQM4M [[10]](#10) molecular dataset, where each node is an atom, and the edges correspond to the molecular structure. The task is to predict pairs of nodes that will be less than 3.5 angstroms apart in the final configuration of the molecule. To ensure that only ‘long-range’ predictions are counted, the task is limited to pairs of molecules that are separated by at least 5 hops.
4. __Peptides-func__ and __Peptides-struct__: these are derived from the SATPdb [[11]](#11) dataset of peptides, a class of molecules that is characterised by a large number of nodes and complex structure. While typical peptide datasets use nodes to represent amino acids, the authors instead split these into multiple nodes, each representing individual atoms. In doing so, they imposed extra separation between the graphs. Then they defined two tasks, one a graph-level regression, one a graph-level classification, to predict molecular properties of the graph.


## Are these truly ‘long range’ benchmarks?

The central claim of the paper is that the above datasets provide a benchmark for assessing whether a new method solves the LRI problem.
In this section, we describe the arguments that the authors make in support of this claim, and discuss their strengths and weaknesses.

TODO add a bit more here about what a good benchmark should look like.
 

### LRI by Construction

The authors argue that they construct their datasets in such a way that acheiving good performance on them requires solving the LRI problem. In one cases, this is convincing: the PCQM contact dataset only considers interactions between distant nodes, and so cannot be solved by local information alone.

In the other cases, the justification is murkier. For example, they argue that classifying superpixels in the COCO and Pascal datasets is inherently long range, but they provide no argument for why this is the case.

Similarly, they argue that because 3D folding structure is important in determining peptide properties, and because 3D structures determine . However, it's not clear whether how this dependence on 3D geometry can be understood in terms of over-squashing and over-smoothing, which have only been characterised as dependent on the graph topology. 

In summary, there isn't a strong _a priori_ reason to believe that any of the datasets are characterised by LRI, except for PCQM.

### Relative outperformance of transformer methods

The authors showed that transformer architectures, which ignore the input graph in favour of a fully connected one, outperformed other methods in 4 out of the 5 datasets. While they interpreted this as evidence of LRI in the data, there are other plausible explanations. For example, it’s possible that the extra expressivity afforded by the transformers attention mechanism was responsible for the improved performance. 

TODO check whether expressivity this was also reflected in generalisation statistics.

Arguably, we should look for more direct evidence that improved performance was due to an ability to leverage long range information.


### Graph statistics
The authors generated statistics characterising the graphs found in each dataset, such as the graph diameter, the average shortest path, and the number of nodes. They claimed that high values of these statistics indicated LRI within the graph. However, it’s not clear whether these statistics actually capture features of the graph topology relevant to the LRI problem. A graph may have a large number of nodes, and a complex topology, but it doesn't follow that a task defined on that graph can only be solved by modelling global interactions. For example, consider the case where our task is to calculate the sum of the hidden features in nodes - this isn't dependent on graph topology.

Therefore, we think these graph statistics need to be more directly linked with model performance before we can conclude that they are proof of LRI in the datasets.

 


### Importance of Positional Encodings
TODO


### Our Paper

Our project attempted to address the weaknesses we identified each of the above arguments. Our ultimate goals were:

1. To replicate the results of the original study. 
2. to give our reader greater confidence that these datasets are a suitable benchmark for LRI methods, in the sense that improvements on these benchmarks can be attributed to an increased ability to solve the LRI problem.
3. to provide a better characterisation of which LRI factors (e.g. oversquashing or oversmoothing) were important.

Because we had limited computational resources, we chose to focus on the Pascal dataset. Because this is a node classification dataset, it allows us to investigate long range interactions in ways that are impossible for graph level tasks.

# 3. Experiments

### Which models perform best on the Pascal Dataset?

We began by training a number of models on the Pascal VOC dataset. This both replicated the original models, and gave us access to a set of models that we could use to test hypotheses about the presence of LRI in the dataset.

As with the original study, we found that a transformer architecture performed better than all other models. However, we also tested a variety of MPNN models that explicitly encoded geometric information. We felt that these were a 'fairer' test of the capacity of a message passing network, because the geometric relationship between two nodes is more semantically meaningful than the one imposed by the arbitrary topology of the superpixel boundary graph. We found that these models gave comparable performance  to the transformer, even with as few as two message passing layers.

A brief description of the models, and their performance, is given below:

#TODO need to format this better. Remove # params in favour of number of layers. Change to 2 s.f. Colour code.
#TODO add column describing models?
    | Model                   | # Params  | Best train F1  | Best val F1 | Best test F1 |
|-------------------------|-----------|----------------|-------------|--------------|
| GCN                     | 496k      | 0.46046        | 0.15339     | 0.1585       |
| E(n)-Invariant          | 523k      | 0.44664        | 0.21416     | 0.2213       |
| E(n)-Invariant (JK 1)   | 572k      | 0.38194        | 0.22385     | 0.23597      |
| E(n)-Invariant (JK 2)   | 523k      | 0.51587        | 0.23583     | 0.23675      |
| E(n)-Equivariant        | 523k      | 0.3767         | 0.2434      | 0.2516       |
| E(n)-Equivariant (JK 1) | 572k      | 0.4502         | 0.2431      | 0.2494       |
| E(n)-Equivariant (JK 2) | 523k      | 0.4613         | 0.2399      | 0.2453       |
| Transformer+LapPE       | 501k      | 0.8062         | 0.2624      | 0.2610       |

Recall that our second goal above was to see whether improvements on the LRGB were caused by an improved ability to model long range interactions. From this point of view, these results are worrisome, because we found that a model that could only use local information - which is by definition not capable of modelling LRI - was nearly as performant as one that could model interactions between all nodes.

TDOO more discussion of the JK models? These are interesting because they are explicitly done to help oversmoothing.

### Is performance correlated with increased importance of distant nodes

If the Pascal dataset was truly characterised by LRI, we should expect two things:
- for models that treat distant nodes the same way as nearby ones (like transformers), we expect that the features of those distant nodes are important to the accuracy of their predictions.
- for MPNN architectures that can't make use of interactions, we expect that

To test these hypotheses, we used __influence functions__ (REF) to quantify the importance of nodes at different. Briefly, the influence function is defined as

We additionally 

### Does model performance correlate with graph statistics?

### Does rewiring the graph to remove bottlenecks improve performance?


## Conclusions











We summarise our main contributions as follows:
1. We apply curvature-based rewiring method -  Stochastic Discrete Ricci Flow (SDRF) algorithm to Pascal SP [[2]](#2). As shown in [Figure 2](#fig2), we rewire the graph and pass it through trained graph models. The aim is to analyse whether rewiring helps to mitigate the problem of oversquashing.  
2. Pascal-SP dataset has node embedding of size 14 where the first 12 features are related to the color of the node and the last 2 features are related to its x,y position. We separate the feature embedding into node embedding and coordinate embedding, to incorporate type-0 representations (i.e. relative distances between the nodes). We do this, inorder to see whether considering geometrical representations of the nodes can help with oversquashing. We implement E(n)-invariant and E(n)-equivariant graph networks [[3]](#3) to see whether these 2 variants of translation equivariance architectures help to handle oversquashing. [Figure 1](#fig1) presents an overview of the architectures used. 
3. To handle the issue of oversmoothing where the values of hidden node converge in deeper layers, we apply some of the "jumping techniques" introduced in [[4]](#4).  Specifically, we apply concatenation and max pooling of all the layers in every forward pass. 
4. We measure the sensitivity of node x to node y, or the influence of y on x, by measuring how much a change in the input feature of y affects the representation of x in the last layer. This helps us to see how for different models, the influences over short and long graph distances change. We implement the definition of this influence score introduced in Section 3, Definition 3.1 of [[4]](#4).
5. We analyse the usage of the Cheeger constant [[2]](#2) and minimal average path as "measurements of LRI" in the graph by attempting to find a correlation between the values of these metrics and the level of influence of long-range interactions on different models' prediction.
We predict that high Cheeger values would correlate with high bottleneck and high average shortest path would correlate with the range of interactions. As the oversquashing problem relates to an interaction between bottlenecking and distance, we predict that models that should perform better under LRI tasks would specifically perform better on
graphs with high values on both of these metrics.
6. Finally advanced  models like Steerable GNN [[5]](#5) have been used mostly in toy datasets like N-body and QM9. We experiment this with Pascal-SP to verify whether steerable messages improve upon the above mentioned trivial equivariant graph networks that send invariant messages. Ideally we expect this message passing approach, should be maximally expressive due to E(3) equivariance. (This is an ongoing Work)


As part of contribution #5, we compute the average shortest path using the [networkx implementation](https://networkx.org/documentation/networkx-1.3/reference/generated/networkx.average_shortest_path_length.html).
For the Cheeger metric we use the Cheeger constant inequality to compute the upper and lower bounds of the squared value of the constant and then take the median of these two results[6].
We do not compute the Cheeger metric directly because of the computational cost of computing the edge boundary.




| <img width="1246" alt="image" src="https://github.com/madhurapawaruva/uva-dl2-team11-forpeer/assets/117770386/9b0c9463-008f-47b7-817a-9a63c796e8a7" id="fig1">    | <img width="739" alt="image" src="https://github.com/madhurapawaruva/uva-dl2-team11-forpeer/assets/117770386/ed650fa6-ec70-4c9f-9594-87bcddc3cff2" id="fig2"> | 
| -------- | -------- |
| Figure 1: E(n)-Invariant and E(n)-Equivariant Architecture    | Figure 2: Rewiring Inference Architecture  |  
  
  | <img src="assets/cheeger-median-against-path.png"> | <img src="assets/heat-map-three-values.png"> | <img src="assets/histogram-graph-values-destribution.png"> |
 | -------------------------------------------|-------------------------------|-----------------------------------------------------------|
 | Figure 3: Cheeger value against average shortest path | Figure 4: Heat Map of graph diameter against Cheeger value and average shortest path | Figure 5: Distribution of graphs across the 3 metrics |
# Results

Applying SDRF rewiring to the graphs of Pascal dataset and then training the Transformer+LapPE model gives an improved performance as shown in the table below. We are working on applying the same for other models.

| No. of edge additions | Best test F1 |
| ------------------- | ------------ |
| 0 (original graphs) | 0.261        |
| 10                  | 0.2757       |
| 20                  | 0.2635       |

We see an increase in f1 scores on adding 10 edges to every graph, but we also see a decrease in score on adding 20 edges. It would be interesting to experiment with the amount of edges being added and the effect it has on f1 scores to reach an appropriate threshold after which rewiring becomes detrimental (ongoing work).

We only experiment with the GCN and Transformer+LapPE models from the original paper and stick to the PascalVOC dataset due to time and computational constraints. For a uniform comparison of performance across models, we follow the convention of limiting the number of parameters to approximately 500k. Also, for training of graph models, we switched to using the Cosine learning rate scheduler rather than the default Reduce on Plateau scheduler because of implementation errors we faced on loading the trained checkpoints due to a missing module of the scheduler within torch.

In the table below, we present the F1 scores for the models we trained. Here JK1 denotes the jumping knowledge variant 1 where we concatenate hidden outputs of all layers. And JK2 denotes the jumping knowledge variant where we do maximum pooling of all the layers.

| Model                   | # Params  | Best train F1  | Best val F1 | Best test F1 |
|-------------------------|-----------|----------------|-------------|--------------|
| GCN                     | 496k      | 0.46046        | 0.15339     | 0.1585       |
| E(n)-Invariant          | 523k      | 0.44664        | 0.21416     | 0.2213       |
| E(n)-Invariant (JK 1)   | 572k      | 0.38194        | 0.22385     | 0.23597      |
| E(n)-Invariant (JK 2)   | 523k      | 0.51587        | 0.23583     | 0.23675      |
| E(n)-Equivariant        | 523k      | 0.3767         | 0.2434      | 0.2516       |
| E(n)-Equivariant (JK 1) | 572k      | 0.4502         | 0.2431      | 0.2494       |
| E(n)-Equivariant (JK 2) | 523k      | 0.4613         | 0.2399      | 0.2453       |
| Transformer+LapPE       | 501k      | 0.8062         | 0.2624      | 0.2610       |

We obtain a test f1 score of ~0.16 on the GCN model and use this as a baseline for comparison. The E(n)-Invariant GNN model achieves a higher f1 score which is again improved by concatinating/max pooling layer outputs. The E(n)-Equivariant GNN model further obtains better f1 scores, but concatinating/max pooling the layer outputs does not seem to help in this case. The best results are observed for the Transformer+LapPE model.

Results of Influence Score Distribution: Not Available. We have implemented the function for it but we are still debugging and ensuring that it works right but the result would look something like this: 

<img width="584" alt="image" src="https://github.com/madhurapawaruva/uva-dl2-team11-forpeer/assets/117770386/0e9574c5-4068-4143-b5f7-28546e77b0a1">

This was for E(n)-invariant model and we see how the score differs with the distance. 

We plot [Figure 3](#fig3) by selecting a few graphs from the datasets and observe whether they suffer from long range interactions and bottlenecks. We also generate a heat map similar to [Figure 4](#fig4) with diameter per model
in order to observe the relation between the metrics and the model behaviors. This is just an initial introductory visualization of these metrics. We plan to compare these metrics with the F1-scores generated for various models we discussed in the table above. 

# 4. Conclusion
To conclude, we see quite promising results by using invariant and equivariant graph neural networks. What we are investigating currently is whether this improvement over GCN is due to the ability of these architectures to model Long Range Interactions more effectively. We hope to gain insight regarding this by comparing the aforementioned metrics wrt the F1 scores of models for every graph. 
We also wish to do a more drilled-down analysis of rewiring of graphs and how it affects the performance of the models during inference. Furthermore, we plan to also incorporate steerable GCN to see whether adding more higher representations help us to model LRIs. We also plan to analyse the influence scores of nodes wrt distant nodes generated by various models, to see how sensitive the nodes are to changes in each other’s representations.
To summarise, we have an initial brief results and working code modules now. What remains for us is to dive deep down on analyzing if long range interactions are really and truly modeled by the various alternative approaches we employ by using the various metrics we introduce. 


Note: You might notice some gaps within the blogpost. This is because we do not have all results yet and the work is in progress. We hope to submit our notebook with final results during the main submission since our results and data are scattered and getting generated too. 

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

<a id="6">[6]</a>
Ravi Montenegro and Prasad Tetali "Mathematical Aspects of Mixing Times in Markov Chains"

<a id="7">[7]</a>
Li, Qimai et al. “Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning.” AAAI Conference on Artificial Intelligence (2018).

<a id="8">[8]</a>
Everingham, Mark et al. “The Visual Object Classes (VOC) Challenge.” International Journal of Computer Vision 88 (2010): 303-338.

<a id="9">[9]</a>
Lin, Tsung-Yi et al. “Microsoft COCO: Common Objects in Context.” European Conference on Computer Vision (2014).

<a id="10">[10]</a>
Hu, Weihua et al. “OGB-LSC: A Large-Scale Challenge for Machine Learning on Graphs.” ArXiv abs/2103.09430 (2021): n. pag.

<a id="11">[11]</a>
Singh, Sandeep et al. “SATPdb: a database of structurally annotated therapeutic peptides.” Nucleic Acids Research 44 (2015): D1119 - D1126.

<a id="12">[12]</a>
Alon, Uri, and Eran Yahav. "On the bottleneck of graph neural networks and its practical implications." arXiv preprint arXiv:2006.05205 (2020).