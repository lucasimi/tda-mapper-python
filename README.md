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

from mapper.core import *
from mapper.cover import *
from mapper.clustering import *
from mapper.plot import *

iris_data = load_iris()
iris_data
X = iris_data.data[:, :]
y = iris_data.target

cover_algo = BallCover(radius=1.0, metric=lambda x, y: np.linalg.norm(x - y))
mapper_algo = MapperAlgorithm(cover=cover_algo, clustering=TrivialClustering())
mapper_graph = mapper_algo.build_graph(X)
mapper_plot = MapperPlot(X, mapper_graph)

fig1 = mapper_plot.with_colors().plot('plotly', 512, 512, 'mean lens')
fig1.show()

fig2 = mapper_plot.with_colors(colors=list(y)).plot('plotly', 512, 512, 'mean lens')
fig2.show()

```
![The mapper graph of a random dataset](/examples/graph.png)

### Features

- [x] Topology
    - [x] Any custom lens
    - [x] Any custom metric
- [x] Clustering algoritms
    - [x] Any sklearn clustering algorithm
    - [x] Skip clustering
- [x] Cover algorithms:
    - [x] Ball Cover
    - [x] Knn Cover
    - [ ] Induced Cover

