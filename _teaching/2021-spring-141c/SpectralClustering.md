---
title: "Spectral Clustering"
collection: teaching
permalink: /teaching/2021-spring-141c/SpectralClustering
---



# STA 141C Big-data and Statistical Computing

## Discussion 4: Spectral Clustering

TA: Tesi Xiao

### Graph

A graph is made up of vertices (nodes/points) which are connected edges (links/lines). Mathematically speaking, a graph is an ordered pair $G=(V, E)$

- $G = (V, E)$
- $V$: the set of vertices
- $E\subseteq \{ (x,y)\| x,y\in V \text{ and } x\neq y\}$ (for simplicity, we do not consider self-loops here)
  + Undirected graph: $(x, y)$ is an unordered pair;
  + Directed graph: $(x, y)$ is an ordered pair.
- Weighted graph $G = (V, E, W)$: A weighted graph (or a network) is a graph with weighted edges. That is to say, a number (the weight) $w_{ij}\in W$ is assigned to the edge $(i,j)\in E$. $w_{ij}=0$ if $(i,j)\notin E$. 
  + $|V|$: the cardinality of set $V$ / the number of vertices
  + $|E|$: the cardinality of set $E$ / the number of edges
  + **Weight Matrix** $W$: a $|V|\times |V|$ matrix where $W_{i,j} = w_{ij}$ if $(i,j) \in E$ else $W_{i,j} = 0$. For undirected graphs, $W$ is symmetric, i.e., $w_{ij} = w_{ji}$.

Note that for unweigted graph, one can assign 1 to all the edges, then $W$ would become the **adjacency matrix** $A$
where $A_{i,j} = 1$ if $(i,j)\in E$ else $A_{i,j}=0$ 
    

- **Degree**: For undirected graphs, the degree of a vertix $v\in V$ denoted by $\text{deg}(v)$ is defined as $\text{deg}(v) = \sum_{j\in V} w_{vj}$. For directed graphs, we can similarly define the in-degree and the out-degree of a vertex. See [Wikipage](https://en.wikipedia.org/wiki/Directed_graph).

 Commonly, we use $D$ to represent the **degree matrix**, which is a diagonal matrix including the degrees of vertices.

-  **Laplacian Matrix** for weighted graphs
   + Laplacian Matrix $L$: $L=D-W$
   + Symmetric Normalized Laplacian $\overline{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}$



### Netwokx

[NetworkX](https://networkx.org/) is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. Check out [Tutorials](https://networkx.org/documentation/stable/tutorial.html)


```python
# a simple example of using networkx

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_nodes_from([1,2,3])
G.add_node(4)

G.add_edges_from([(1,2), (2,3), (1,3)])
G.add_edge(3,4)
```


```python
nx.draw(G, with_labels=True)
```



### Graph Cut

- Partitioning into two clusters

  + Naive Balanced Cut:    $$\min_{V_1, V_2} \text{ cut}(V_1, V_2) := \sum_{v_i \in V_1, v_j \in V_2} W_{ij} \text{  s.t.  } |V_1| = |V_2| \text{ and } V_1 \cup V_2 = V, V_1 \cap V_2 = \emptyset$$ 
  + Ratio-Cut:   $$\min_{V_1, V_2} \text{ RC}(V_1, V_2) := \bigg\{ \frac{\text{cut}(V_1, V_2)}{|V_1|} + \frac{\text{cut}(V_1, V_2)}{|V_2|} \bigg\}$$
  + Normalized-Cut:   $$\min_{V_1, V_2} \text{ NC}(V_1, V_2) := \bigg\{ \frac{\text{cut}(V_1, V_2)}{\text{deg}(V_1)} + \frac{\text{cut}(V_1, V_2)}{\text{deg}(V_2)} \bigg\}$$

  In the orginal paper about Normalized-Cut, they call $\text{deg}(V_1)$ as $\text{assoc}(V_1, V)$, which is the total connections from nodes in $V_1$ to all nodes in the graph. See [paper](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf).

- Generalization to $k$ clusters

  + Ratio-Cut:   $$\min_{V_1,..., V_k} \sum_{c=1}^k\frac{\text{cut}(V_c, V-V_c)}{\|V_c\|}  $$
  + Normalized-Cut:   $$\min_{V_1,..., V_k} \sum_{c=1}^k\frac{\text{cut}(V_c, V-V_c)}{\text{deg}(V_c)} $$


### Spectral Clustering

The above graph-cut problems are hard to solve. However, fortunately, one can derive relaxed versions of those problems.

- Relaxed to the real-valued [Rayleigh quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient) minimization problem
  + Ratio-Cut:   $$\min_{y_1,..., y_k} \sum_{c=1}^k\frac{y_c^\top L y_c}{y_c^\top y_c} $$
  + Normalized-Cut:   $$\min_{y_1,..., y_k} \sum_{c=1}^k\frac{y_c^\top \overline{L} y_c}{y_c^\top y_c} $$
  + Remark: $y_c$ is a vector indicates which vertices belong to the $c$-th cluster, thus satifying certain constraints. Putting these $y_c$s together into a matrix, we will get a problem in the form of
    $$\min_{Y^\top Y = I} = \text{Trace}(Y^\top L Y) \text{(Ratio-Cut)   or    }\text{Trace}(Y^\top \overline{L} Y) \text{(Normalized-Cut)}$$
    Then we are able to solve it by finding eigenvectors corresponding the smallest $k$ eigenvalues of $L$. However, the obtained $Y$ may not satisfy the contraints that $Y$ should exactly indicate the partition. We run K-means algorithm on the rows of $Y$ to obtain the final results.

- Summary: We convert the original problem into a eigenvalue problem of the graph Laplacian.

### Spectral Clustering by scikit learn

- [sklearn.cluster.spectral_clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.spectral_clustering.html#sklearn.cluster.spectral_clustering)
- [An example in image segmentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html#sphx-glr-auto-examples-cluster-plot-segmentation-toy-py)
