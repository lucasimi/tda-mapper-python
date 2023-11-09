# tda-mapper-python 

![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg) [![codecov](https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R)](https://codecov.io/github/lucasimi/tda-mapper-python)

In recent years, an ever growing interest in **Topological Data Analysis** (TDA) emerged in the field of data science. The core principle of TDA is to gain insights from data by using topological methods, as they show good resilience to noise, and they are often more stable than many traditional techniques. This Python package provides an implementation of the **Mapper Algorithm**, one of the most common tools from TDA. 

The mapper algorithm takes any dataset $X$ (usually high dimensional), and returns a graph $G$, called **Mapper Graph**. Surprisingly enough, despite living in a 2-dimensional space, the mapper graph $G$ represents a reliable summary for the shape of $X$ (they share the same number of connected components). This feature makes the mapper algorithm a very appealing choice over more traditional approaches, for example those based on projections, because they often give you no way to control shape distortions. Moreover, preventing artifacts is especially important for data visualization: the mapper graph is often a capable tool, which can help you identify hidden patterns in high-dimensional data.

## Basics

Here we'll give just a brief description of the core ideas around the mapper, but the interested reader is advised to take a look at the original [paper](https://research.math.osu.edu/tgda/mapperPBG.pdf). The Mapper Algorithm follows these steps:

1. Take any *lens* you want. A lens is just a continuous map $f \colon X \to Y$, where $Y$ is any parameter space, usually having dimension lower than $X$. You can think of $f$ as a set of KPIs, or features of particular interest for the domain of study. Some common choices for $f$ are *statistics* (of any order), *projections*, *entropy*, *density*, *eccentricity*, and so forth.

![Step 1](https://raw.githubusercontent.com/lucasimi/tda-mapper-python/main/resources/mapper_1.png)

2. Build an *open cover* for $f(X)$. An open cover is a collection of open sets (like open balls, or open intervals) whose union makes the whole image $f(X)$, and can possibly intersect.

![Step 2](https://raw.githubusercontent.com/lucasimi/tda-mapper-python/main/resources/mapper_2.png)

3. For each element $U$ of the open cover of $f(X)$, let $f^{-1}(U)$ be the preimage of $U$ under $f$. Then the collection of all the $f^{-1}(U)$'s makes an open cover of $X$. At this point, split every preimage $f^{-1}(U)$ into clusters, by running any chosen *clustering* algorithm, and keep track of all the local clusters obtained. All these local clusters together make a *refined open cover* for $X$.

![Step 3](https://raw.githubusercontent.com/lucasimi/tda-mapper-python/main/resources/mapper_3.png)

4. Build the mapper graph $G$ by taking a node for each local cluster, and by drawing an edge between two nodes whenever their corresponding local clusters intersect.

![Step 4](https://raw.githubusercontent.com/lucasimi/tda-mapper-python/main/resources/mapper_4.png)

N.B.: The choice of the lens $f$ has a deep practical impact on the mapper graph. Theoretically, if clusters were able to perfectly identify connected components (and if they were "reasonably well behaved"), chosing any $f$ would give the same mapper graph (see the [Nerve Theorem](https://en.wikipedia.org/wiki/Nerve_complex#Nerve_theorems) for a more precise statement). In this case, there would be no need for a tool like the mapper, since clustering algorithms would provide a complete tool to understand the shape of data. Unfortunately, clustering algorithms are not that good. Think for example about the case of $f$ being a constant function: in this case computing the mapper graph would be equivalent to performing clustering on the whole dataset. For this reason a good choice for $f$ would be any continuous map which is somewhat *sensible* to data: the more sublevel sets are apart, the higher the chance of a good local clustering.

## Installation

First, clone this repo, `cd` into the local repo, and install via `pip` from your local repo
```
python -m pip install .
```

## How to use this package - A First Example

In the following example, we use the mapper to perform some analysis on the famous Iris dataset. This dataset consists of 150 records, having 4 numerical features and a label which represents a class. As lens, we chose the PCA on two components. 

```python
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import matplotlib

from tdamapper.core import *
from tdamapper.cover import *
from tdamapper.clustering import *
from tdamapper.plot import *

iris_data = load_iris()
X, y = iris_data.data, iris_data.target
lens = PCA(2).fit_transform(X)

cover = GridCover(n_intervals=7, overlap_frac=0.25)
clustering = AgglomerativeClustering(n_clusters=2, linkage='single')

mapper_algo = MapperAlgorithm(cover, clustering)
mapper_graph = mapper_algo.fit_transform(X, lens)
mapper_plot = MapperPlot(X, mapper_graph)
colored = mapper_plot.with_colors(colors=list(y), agg=np.nanmedian)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
colored.plot_static(title='class', ax=ax)
```

![The mapper graph of the iris dataset](https://raw.githubusercontent.com/lucasimi/tda-mapper-python/main/resources/iris.png)

As you can see from the plot, we can identify two major connected components, one which corresponds precisely to a single class, and the other which is shared by the other two classes.

## A Second Example

In this second example we try to take a look at the shape of the digits dataset. This dataset consists of less than 2000 pictures of handwritten digits, represented as dim-64 arrays (8x8 pictures)

```python
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from tdamapper.core import *
from tdamapper.cover import *
from tdamapper.clustering import *
from tdamapper.plot import *

import matplotlib

digits = load_digits()
X, y = [np.array(x) for x in digits.data], digits.target
lens = PCA(2).fit_transform(X)

cover = GridCover(n_intervals=15, overlap_frac=0.25)
clustering = KMeans(10, n_init='auto')

mapper_algo = MapperAlgorithm(cover, clustering)
mapper_graph = mapper_algo.fit_transform(X, lens)
mapper_plot = MapperPlot(X, mapper_graph, iterations=100)

fig = mapper_plot.with_colors(colors=y, cmap='jet', agg=np.nanmedian).plot_interactive_2d(title='digit', width=512, height=512)
fig.show(config={'scrollZoom': True})
```

![The mapper graph of the digits dataset](https://raw.githubusercontent.com/lucasimi/tda-mapper-python/main/resources/digits.png)

As you can see the mapper graph shows interesting patterns. Note that the shape of the graph is obtained by looking only at the 8x8 pictures, discarding any information about the actual label (the digit). You can see that those local clusters which share the same labels are located in the same area of the graph. This tells you (as you would expect) that the labelling is *compatible with the shape of data*.
Moreover, by zooming in, you can see that some clusters are located next to others. For example in the picture you can see the details of digits '4' (cyan) and '7' (red) being located one next to the other.

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
