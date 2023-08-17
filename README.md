# tda-mapper-python 

![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg)

In recent years, an ever growing interest in **Topological Data Analysis** (TDA) emerged in the field of data science. The core principle of TDA is to rely on topological methods to gain valuable insights from datasets, as topology provides tools which are more robust to noise than many more traditional techniques. This Python package provides an implementation of the **Mapper Algorithm** from TDA. The mapper algorithm takes any dataset $X$ (in any dimension), and returns a graph $G$, called **Mapper Graph**. Despite living in a 2-dimensional space, the mapper graph $G$ represents a reliable summary for the shape of $X$, and, more importantly, they have the same number of connected components. This feature makes the mapper algorithm a very appealing choice over more traditional approaches based on projections, as they often offer low to no control on how the shape gets distorted. This is especially important when you want to visualize the shape of a dataset: using the wrong tool it's easy to lose relationships across subsets of data, but the mapper algorithm can reduce the effects of distorsions.

## Basics

Here we'll give just a brief description of the core ideas around the mapper, but the interested user is advised to take a look at the original [paper](https://research.math.osu.edu/tgda/mapperPBG.pdf). The Mapper Algorithm follows these steps:

1. Take any *lens* you want. A lens is just a continuous map $f \colon X \to Y$, where $Y$ is any parameter space, usually with dimension lower than $X$. You can think about $f$ as a set of KPIs, or features of particular interest for the domain of study. Some common choices for $f$ are *statistics* (of any order), *projections*, *entropy*, *density*, *eccentricity*, and so forth.

2. Build an *open cover* for $f(X)$. An open cover is a collection of open sets (like open balls, or open intervals) whose union makes the whole image $f(X)$, and can possibly intersect.

3. For each open set $U$ of $f(X)$ let $V$ be the preimage of $U$ under $f$. Then the collection of $V$'s makes an open cover of $X$. For each $V$, run any chosen *clustering* algorithm and keep track of all the local clusters. Here we use clustering as the statistical version of the topological notion of connected components.

4. Build the mapper graph $G$, by taking a node for each local cluster, and by drawing an edge between two nodes whenever their corresponding clusters intersect.

N.B.: The choice of the lens $f$ has a deep practical impact on the mapper graph. Theoretically, if clusters were able to perfectly catch connected components (and if they were "reasonably well behaved"), chosing any $f$ would give the same mapper graph (see the Nerve Theorem for a more precise statement). In this case, there would be no need for a tool like the mapper, since clustering algorithms would provide a more complete tool to understand the shape of data. Unfortunately, clustering algorithms are not perfect. As an example for how important $f$ is, think about the case of $f$ being a constant function. In this setting the preimage of any open cover would be the whole dataset $X$, then computing the mapper graph would be equivalent to performing clustering on $X$. For this reason a good choice for $f$ would be any continuous map which is somewhat *sensible* to data: the more sublevel sets show a clean clustered behavior, the higher the chance of obtaining a good local clustering. In practice, chosing the right lens is much easier than chosing the right metric (or the right clustering algorithm), but can be quite tricky too. For this reason, the mapper algorithm is an interactive tool which is expected to be tuned by looking at the mapper graph, after some trials.

## How to use this package - A First Example

First, clone this repo, and install via pip `python -m pip install .`. 

In the following example, we use the mapper to perform some analysis on the famous Iris dataset. This dataset consists of 150 records, having 4 numeric features and a label which represents a class. As a lens we chose the PCA on two components. 

```python

from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from mapper.core import *
from mapper.cover import *
from mapper.clustering import *
from mapper.plot import *

import matplotlib

iris_data = load_iris()
X, y = iris_data.data, iris_data.target
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(
    cover=CubicCover(n=10, perc=0.5), 
    clustering=AgglomerativeClustering(n_clusters=None, linkage='single'))
mapper_graph = mapper_algo.fit_transform(X, lens)
mapper_plot = MapperPlot(X, mapper_graph)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
colored = mapper_plot.with_colors(colors=list(y), agg=np.nanmedian)
colored.plot_static(title='class', ax=ax)

```
![The mapper graph of the iris dataset](/examples/iris.png)

## A Second Example

In this second example we try to take a look at the shape of the digits dataset. This dataset consists of less than 2000 pictures of handwritten digits, represented as dim-64 arrays (8x8 pictures)

```python
from sklearn.datasets import load_digits
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA

from mapper.core import *
from mapper.cover import *
from mapper.clustering import *
from mapper.plot import *

import matplotlib

digits = load_digits()
X, y = [np.array(x) for x in digits.data], digits.target
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(cover=CubicCover(n=10, perc=0.25), clustering=KMeans(10, n_init='auto'))
mapper_graph = mapper_algo.fit_transform(X, lens)
mapper_plot = MapperPlot(X, mapper_graph, iterations=100)
mapper_plot.with_colors(colors=y, cmap='jet', agg=np.nanmedian).plot_interactive_2d(width=512, height=512)

```
![The mapper graph of the digits dataset](/examples/digits.png)

As you can see, the mapper can give you an interesting visual feedback about what's going on.

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
