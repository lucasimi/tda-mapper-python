"""
This library provides a Python implementation of the TDA Mapper algorithm,
which is used for topological data analysis (TDA). The TDA Mapper algorithm
is a method for extracting topological features from high-dimensional data
by constructing a simplicial complex that captures the shape of the data.

The `tdamapper` package includes the following main modules:
- `core`: Contains the core implementation of the Mapper algorithm.
- `cover`: Provides classes for creating _open covers_, which are collections of overlapping sets
    that cover the data space.
- `learn`: Includes classes compatible with scikit-learn's estimator API based on Mapper. These
    classes can be used in scikit-learn pipelines.
- `utils`: Provides utility functions for creating spatial indexes.
- `plot`: Contains functions for visualizing the Mapper graph.

To use the TDA Mapper algorithm, you can follow these steps:

Examples
--------
>>> from sklearn.datasets import make_circles
>>>
>>> import numpy as np
>>> from sklearn.decomposition import PCA
>>> from sklearn.cluster import DBSCAN
>>>
>>> from tdamapper.learn import MapperAlgorithm
>>> from tdamapper.cover import CubicalCover
>>> from tdamapper.plot import MapperPlot
>>>
>>> X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)
>>> y = PCA(2, random_state=42).fit_transform(X)
>>>
>>> cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
>>> clust = DBSCAN()
>>> graph = MapperAlgorithm(cover, clust).fit_transform(X, y)
>>>
>>> fig = MapperPlot(graph, dim=2, seed=42, iterations=60).plot_plotly(colors=labels)
>>> fig.show(config={"scrollZoom": True})
"""
