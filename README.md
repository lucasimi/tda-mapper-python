# mapper-tda 

![test](https://github.com/lucasimi/mapper-tda/actions/workflows/test.yml/badge.svg)

This library is an implementation of the Mapper Algorithm from Topological Data Analysis, a branch of data analysis using topological tools to recover insights from datasets. In the following, we give a brief description of the algorithm, but the interested user is advised to take a look at the original [paper](https://research.math.osu.edu/tgda/mapperPBG.pdf). The Mapper Algorithm builds a graph from a given dataset, and some user choices. The output graph, called "mapper graph" gives a global approximation of (some) topological features of the original dataset, giving direct information about its shape and its connected components.

### Input

Assume we have a dataset D inside a metric space X, together with the following choices:

1. A continuous map f:X -> Y 
2. A cover algorithm for f(D)
3. A clustering algorithm for D.

### Steps

The mapper algorithm follows these steps:

1. Build an open cover of f(D)
2. For each open chart U of f(D) let V the preimage of U under f, then the V's form an open cover of D. For each V, run the chosen clustering algorithm
3. For each local cluster obtained, build a node. Whenever two local clusters (from different V's) intersect, draw an edge between their corresponding nodes.

The graph obtained is called a "mapper graph".

## How to use this library

First, clone this repo, and install this library via pip install `python -m pip install .`. In the following example, available [here](examples/example_notebook.ipynb), we compute the mapper graph on a random dataset, using the identity lens and the euclidean metric. The clustering algorithm can be any class implementing a `fit` method, as [`sklearn.cluster`](https://scikit-learn.org/stable/modules/clustering.html) algorithms do, and returning an object which defines a `.labels_` field.

```python
import numpy as np

from mapper.cover import SearchCover
from mapper.search import BallSearch
from mapper.pipeline import MapperPipeline
from mapper.network import Network

from sklearn.cluster import DBSCAN

mp = MapperPipeline(
    cover_algo=SearchCover(search_algo=BallSearch(1.5), 
                           metric=lambda x, y: np.linalg.norm(x - y), 
                           lens=lambda x: x),
    clustering_algo=DBSCAN(eps=1.5, min_samples=2)
    )

data = [np.random.rand(10) for _ in range(100)]
g = mp.fit(data)
nw = Network(g)
nw.plot(data)
```
![The mapper graph of a random dataset](/examples/graph.png)

### Features

- [x] Topology
    - [x] Any custom lens
    - [x] Any custom metric
- [x] Clustering algoritms
    - [x] Any sklearn clustering algorithm
    - [x] No clustering
- [ ] Cover algorithms:
    - [x] Ball Cover
    - [ ] Cubic Cover
    - [x] Knn Cover



