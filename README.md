# tda-mapper-python 

![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg) [![codecov](https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R)](https://codecov.io/github/lucasimi/tda-mapper-python)

In recent years, an ever growing interest in **Topological Data Analysis** (TDA) emerged in the field of data science. The core idea of TDA is to gain insights from data by using topological methods that are proved to be reliable with respect to noise, and that behave nicely with respect to dimension. This Python package provides an implementation of the **Mapper Algorithm**, a well-known tool from TDA. 

The Mapper Algorithm takes any dataset $X$ and returns a *shape-summary* in the form a graph $G$, called **Mapper Graph**. It's possible to prove, under reasonable conditions, that $X$ and $G$ share the same number of connected components.

## Basics

Let $f$ be any chosen *lens*, i.e. a continuous map $f \colon X \to Y$, being $Y$ any parameter space (*typically* low dimensional). In order to build the Mapper Graph follow these steps:

1. Build an *open cover* for $f(X)$, i.e. a collection of *open sets* whose union makes the whole image $f(X)$.

2. Run clustering on the preimage of each open set. All these local clusters together make a *refined open cover* for $X$.

3. Build the mapper graph $G$ by taking a node for each local cluster, and by drawing an edge between two nodes whenever their corresponding local clusters intersect.

To get an idea, in the following picture we have $X$ as an X-shaped point cloud in $\mathbb{R}^2$, with $f$ being the *height function*, i.e. the projection on the $y$-axis. In the leftmost part we cover the projection of $X$ with three open sets. Every open set is represented with a different color. Then we take the preimage of these sets, cluster then, and finally build the graph according to intersections.

![Steps](resources/mapper.png) 

The choice of the lens is the most relevant on the shape of the Mapper Graph. Some common choices are *statistics*, *projections*, *entropy*, *density*, *eccentricity*, and so forth. However, in order to pick a good lens, specific domain knowledge for the data at hand can give a hint. For an in-depth description of Mapper please read [the original paper](https://research.math.osu.edu/tgda/mapperPBG.pdf). 

## Installation

Clone this repo, and install via `pip` from your local directory
```
python -m pip install .
```
Alternatively, you can use `pip` to install directly from GitHub
```
pip install git+https://github.com/lucasimi/tda-mapper-python.git
```
If you want to install the version from a specific branch, for example `develop`, you can run
```
pip install git+https://github.com/lucasimi/tda-mapper-python.git@develop
```

## A worked out example

In order to show how to use this package, we perform some analysis on the the well known dataset of hand written digits (more info [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)), consisting of less than 2000 8x8 pictures represented as arrays of 64 elements.

```python
from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import *
from tdamapper.cover import *
from tdamapper.clustering import *
from tdamapper.plot import *

import matplotlib

digits = load_digits()
X, y = [np.array(x) for x in digits.data], digits.target
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(
    cover=GridCover(n_intervals=10, overlap_frac=0.65),
    clustering=AgglomerativeClustering(10),
    verbose=True,
    n_jobs=8)
mapper_graph = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(X, mapper_graph,
    colors=y, 
    cmap='jet', 
    agg=np.nanmean,
    dim=2,
    iterations=400)
fig_mean = mapper_plot.plot(title='digit (mean)', width=600, height=600)
fig_mean.show(config={'scrollZoom': True})
```

![The mapper graph of the digits dataset, colored according to mean value](resources/digits_mean.png)

It's also possible to obtain a new plot colored according to different values, while keeping the same computed geometry. For example, if we want to visualize how much dispersion we have on each cluster, we could plot colors according to the standard deviation

```python
fig_std = mapper_plot.with_colors(
    colors=y, 
    cmap='viridis', 
    agg=np.nanstd,
).plot(title='digit (std)', width=600, height=600)
fig_std.show(config={'scrollZoom': True})
```

![The mapper graph of the digits dataset, colored according to std](resources/digits_std.png)

The mapper graph of the digits dataset shows a few interesting patterns. For example, we can make the following observations:

* Clusters that share the same color are all connected together, and located in the same area of the graph. This behavior is present in those digits which are easy to tell apart from the others, for example digits 0 and 4.

* Some clusters are not well separated and tend to overlap one on the other. This mixed behavior is present in those digits which can be easily confused one with the other, for example digits 5 and 6.

* Clusters located across the "boundary" of two different digits show a transition either due to a change in distribution or due to distorsions in the hand written text, for example digits 8 and 2.


### Development - Supported Features

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
