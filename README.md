# tda-mapper-python 

![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg)

In recent years, an ever growing interest in **Topological Data Analysis** (TDA) emerged in the field of data analysis. The core principle of TDA is to rely on topological methods to gain reliable insights about datasets, as topology provides tools which are more robust to noise than many more traditional techniques. This Python package provides an implementation of the **Mapper Algorithm** from TDA. In the following, we give a brief description of the core ideas around the mapper, but the interested user is advised to take a look at the original [paper](https://research.math.osu.edu/tgda/mapperPBG.pdf).

The mapper algorithm takes any dataset $X$ (in any dimension), and returns a graph $G$, called **Mapper Graph**. As a 2-dimensional object, the mapper graph represents a reliable summary for the shape of $X$. Surprisingly enough, despite living in a 2-dimensional space, the connected components of $X$ are completely described by the mapper graph, i.e. they share the same number of connected components. This feature makes the mapper algorithm a very appealing choice over more traditional approaches based on projections, as they often offer low to no control on how the shape gets distorted.

The Mapper Algorithm follows these steps:

1. Take any *lens* you want. A lens is just a continuous map $f \colon X \to Y$. You can think about $f$ as a set of KPIs, or features of particular interest for the domain of study. Some common choices for $f$ are *statistics* (of any order), *projections*, *entropy*, *density*, *eccentricity*, and so forth.

2. Build an *open cover* for $f(X)$. An open cover is a collection of open sets (like open balls, or open intervals) whose union makes the whole image $f(X)$, and can possibly intersect.

3. For each open set $U$ of $f(X)$ let $V$ be the preimage of $U$ under $f$. Then the collection of $V$'s makes an open cover of $X$. For each $V$, run any chosen *clustering* algorithm and keep track of all the local clusters. Here we use clustering as the statistical version of the topological notion of connected components.

4. Build the mapper graph $G$, by taking a node for each local cluster, and by drawing an edge between two nodes whenever their corresponding clusters intersect.

N.B.: The choice of the lens $f$ has a deep practical impact on the mapper graph. Theoretically, if clusters were able to perfectly catch connected components (and if they were "reasonably well behaved"), chosing any $f$ would give the same mapper graph (see the Nerve Theorem for a more precise statement). Unfortunately, clustering algorithms are not perfect tools. Think for example about the case of $f$ being a constant function: in this case computing the mapper graph would be equivalent to performing clustering on the whole dataset. For this reason a good choice for $f$ would be any continuous map which is somewhat *sensible* to data: the more sublevel sets are apart, the higher the chance of a good local clustering.

## How to use this package - A Simple Example

First, clone this repo, and install via pip `python -m pip install .`. 

In the following example, we use the mapper to perform some analysis on the famous Iris dataset. This dataset consists of 150 records, having 4 numeric features and a label which represents a class. As a lens we chose the PCA on two components. 

```python
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from mapper.core import *
from mapper.cover import *
from mapper.clustering import *
from mapper.plot import *

import matplotlib

iris_data = load_iris()
X, y = iris_data.data, iris_data.target
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(cover=CubicCover(n=10, perc=0.25), clustering=TrivialClustering())
mapper_graph = mapper_algo.fit_transform(X, lens)
mapper_plot = MapperPlot(X, mapper_graph)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for c in range(3):
    colored = mapper_plot.with_colors(colors=[1 if x == c else 0 for x in list(y)])
    colored.plot(axs[c], 'matplotlib', 512, 512, f'class {c}')
```
![The mapper graph of a random dataset](/examples/graph.png)

### Features

- [x] Topology
    - [x] Any custom lens
    - [x] Any custom metric
- [x] Cover algorithms:
    - [x] Cubic Cover
    - [x] Ball Cover
    - [x] Knn Cover
- [x] Clustering algoritms
    - [x] Any sklearn clustering algorithm
    - [x] Skip clustering
    - [x] Clustering induced by cover
