# tda-mapper

[![PyPI version](https://badge.fury.io/py/tda-mapper.svg)](https://badge.fury.io/py/tda-mapper)
[![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml)
[![deploy](https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml)
[![docs](https://readthedocs.org/projects/tda-mapper/badge/?version=latest)](https://tda-mapper.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R)](https://codecov.io/github/lucasimi/tda-mapper-python) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10642381.svg)](https://doi.org/10.5281/zenodo.10642381)

The Mapper algorithm is a well-known technique in the field of topological data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining, and social sciences, due to its ability to preserve topological features of the underlying space, providing a visual representation that facilitates exploration and interpretation.
For an in-depth coverage of Mapper you can read [the original paper](https://research.math.osu.edu/tgda/mapperPBG.pdf). 

This Python package provides a simple and efficient implementation of Mapper algorithm.

* Installation from package: ```python -m pip install tda-mapper```

* Installation from sources: clone this repo and run ```python -m pip install .```

* Documentation: https://tda-mapper.readthedocs.io/en/latest/ 

## Usage

[Here](https://github.com/lucasimi/tda-mapper-python/raw/main/tests/example.py) you can find a worked out example that shows how to use this package. 
In the example we perform some analysis on the the well known dataset of [hand written digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

```python
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.clustering import FailSafeClustering
from tdamapper.plot import MapperPlot

# We load a labelled dataset
X, y = load_digits(return_X_y=True)
# We compute the lens values
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.65),
    # We prevent clustering failures
    clustering=FailSafeClustering(
        clustering=AgglomerativeClustering(10),
        verbose=False))
mapper_graph = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(
    X, mapper_graph,
    # We color according to digit values
    colors=y,
    # Jet colormap, used for classes
    cmap='jet',
    # We aggregate on graph nodes according to mean
    agg=np.nanmean,
    dim=2,
    iterations=50,
    seed=123)
fig_mean = mapper_plot.plot(title='digit (mean)', width=600, height=600)
fig_mean.show(config={'scrollZoom': True})

# We reuse the graph plot with the same positions
fig_std = mapper_plot.with_colors(
    colors=y,
    # Viridis colormap, used for ranges
    cmap='viridis',
    # We aggregate on graph nodes according to std
    agg=np.nanstd,
).plot(title='digit (std)', width=600, height=600)
fig_std.show(config={'scrollZoom': True})
```

| Average class                                                                                                                          | Standard deviation of class                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| ![Mapper graph of digits, colored according to mean](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/digits_mean.png) | ![Mapper graph of digits, colored according to std](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/digits_std.png) |
