# tda-mapper-python 

![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg)

This library is an implementation of the Mapper Algorithm from Topological Data Analysis, a branch of data analysis using topological tools to recover insights from datasets. In the following, we give a brief description of the algorithm, but the interested user is advised to take a look at the original [paper](https://research.math.osu.edu/tgda/mapperPBG.pdf). The Mapper Algorithm builds a graph from a given dataset, and some user choices. The output graph, called "mapper graph" gives a global approximation of (some) topological features of the original dataset, giving direct information about its shape and its connected components.

### Input

Assume we have a dataset D inside a metric space X, together with the following choices:

1. A continuous map $f \colon X \to Y$
2. A cover algorithm for $f(D)$
3. A clustering algorithm for $D$.

### Steps

The mapper algorithm follows these steps:

1. Build an open cover of $f(D)$
2. For each open chart $U$ of $f(D)$ let $V$ the preimage of $U$ under $f$, then the $V$'s form an open cover of $D$. For each $V$, run the chosen clustering algorithm
3. For each local cluster obtained, build a node. Whenever two local clusters (from different $V$'s) intersect, draw an edge between their corresponding nodes.

The graph obtained is called a "mapper graph".

## How to use this library

First, clone this repo, and install this library via pip install `python -m pip install .`. In the following example, available [here](examples/example_notebook.ipynb), we compute the mapper graph on a random dataset, using the identity lens and the euclidean metric. The clustering algorithm can be any class implementing a `fit` method, as [`sklearn.cluster`](https://scikit-learn.org/stable/modules/clustering.html) algorithms do, and returning an object which defines a `.labels_` field.

```python
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

from mapper.core import *
from mapper.cover import *
from mapper.clustering import *
from mapper.plot import *

import matplotlib

iris_data = load_iris()
X, y = iris_data.data, iris_data.target

mapper_algo = MapperAlgorithm(cover=CubicCover(n=10, perc=0.25), clustering=TrivialClustering())
mapper_graph = mapper_algo.build_graph(X)
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

