![Logo](https://github.com/lucasimi/tda-mapper-python/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png)

[![PyPI version](https://img.shields.io/pypi/v/tda-mapper?logo=python&logoColor=silver)](https://pypi.python.org/pypi/tda-mapper)
[![downloads](https://img.shields.io/pypi/dm/tda-mapper?logo=python&logoColor=silver)](https://pypi.python.org/pypi/tda-mapper)
[![codecov](https://img.shields.io/codecov/c/github/lucasimi/tda-mapper-python?logo=codecov&logoColor=silver)](https://codecov.io/github/lucasimi/tda-mapper-python)
[![test](https://img.shields.io/github/actions/workflow/status/lucasimi/tda-mapper-python/test-unit.yml?logo=github&logoColor=silver&branch=main&label=test)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test-unit.yml)
[![publish](https://img.shields.io/github/actions/workflow/status/lucasimi/tda-mapper-python/publish-pypi.yml?logo=github&logoColor=silver&label=publish)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/publish-pypi.yml)
[![docs](https://img.shields.io/readthedocs/tda-mapper/main?logo=readthedocs&logoColor=silver)](https://tda-mapper.readthedocs.io/en/main/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.10642381-blue?logo=doi&logoColor=silver)](https://doi.org/10.5281/zenodo.10642381)

# tda-mapper
 
**tda-mapper** is a Python library built around the Mapper algorithm, a core
technique in Topological Data Analysis (TDA) for extracting topological
structure from complex data. Designed for computational efficiency and
scalability, it leverages optimized spatial search methods to support
high-dimensional datasets. The library is well-suited for integration into
machine learning pipelines, unsupervised learning tasks, and exploratory data
analysis.

Further details in the
[documentation](https://tda-mapper.readthedocs.io/en/main/)
and in the
[paper](https://openreview.net/pdf?id=lTX4bYREAZ).

### Core Features

- **Efficient construction**
    
    Leverages optimized spatial search techniques and parallelization to
    accelerate the construction of Mapper graphs, supporting the analysis of
    high-dimensional datasets.

- **Scikit-learn integration**

    Provides custom estimators that are fully compatible with scikit-learn's
    API, enabling seamless integration into scikit-learn pipelines for tasks
    such as dimensionality reduction, clustering, and feature extraction.

- **Flexible visualization**

    Multiple visualization backends supported (Plotly, Matplotlib, PyVis) for
    generating high-quality Mapper graph representations with adjustable 
    layouts and styling.

- **Interactive app**

    Provides an interactive web-based interface for dynamic exploration of
    Mapper graph structures, offering real-time adjustments to parameters and
    visualizations.

## Background

The Mapper algorithm extracts topological features from complex datasets,
representing them as graphs that highlight clusters, transitions, and key
structural patterns. These insights reveal hidden data relationships and are
applicable across diverse fields, including social sciences, biology, and
machine learning. For an in-depth overview of Mapper, including its
mathematical foundations and practical applications, read 
[the original paper](https://research.math.osu.edu/tgda/mapperPBG.pdf).

| Step 1 | Step 2 | Step 3 | Step 4 |
| ------ | ------ | ------ | ------ |
| ![Step 1](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_1.png) | ![Step 2](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_2.png) | ![Step 3](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_3.png) | ![Step 2](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_4.png) |
| Choose lens | Cover image | Run clustering | Build graph |

## Quick Start

### Installation

To install the latest version uploaded on PyPI

```bash
pip install tda-mapper
```

### How to Use

Here's a minimal example using the **circles dataset** from `scikit-learn` to
demonstrate how to use **tda-mapper**. This example demonstrates how to apply
the Mapper algorithm on a synthetic dataset (concentric circles). The goal is
to extract a topological graph representation using `PCA` as a lens and
`DBSCAN` for clustering. We proceed as follows:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from tdamapper.learn import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

# Generate toy dataset
X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)
plt.figure(figsize=(5, 5))
plt.scatter(X[:,0], X[:,1], c=labels, s=0.25, cmap="jet")
plt.axis("off")
plt.show()

# Apply PCA as lens
y = PCA(2, random_state=42).fit_transform(X)

# Mapper pipeline
cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
clust = DBSCAN()
graph = MapperAlgorithm(cover, clust).fit_transform(X, y)

# Visualize the Mapper graph
fig = MapperPlot(graph, dim=2, seed=42, iterations=60).plot_plotly(colors=labels)
fig.show(config={"scrollZoom": True})
```

| Original Dataset | Mapper Graph |
| ---------------- | ------------ |
| ![Original Dataset](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_dataset_v2.png) | ![Mapper Graph](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_mean_v2.png) |

Left: the original dataset consisting of two concentric circles with noise,
colored by class label. Right: the resulting Mapper graph, built from the PCA
projection and clustered using DBSCAN. The two concentric circles are well
identified by the connected components in the Mapper graph.

More examples can be found in the
[documentation](https://tda-mapper.readthedocs.io/en/main/examples.html).

## Interactive App

Use our app to interactively visualize and explore your data without writing
code. You can try it right away using 
[our live demo](https://tda-mapper-app.up.railway.app/),
or run it locally on your machine.

To run it locally:

1. Install the app and its dependencies:

    ```bash
    pip install tda-mapper[app]
    ```

2. Launch the app:

    ```bash
    tda-mapper-app
    ```

![tda-mapper-app](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/tda-mapper-app.png)

## Citations

If you use **tda-mapper** in your work, please consider citing both the
[library](https://doi.org/10.5281/zenodo.10642381), archived in a permanent
Zenodo record, and the [paper](https://openreview.net/pdf?id=lTX4bYREAZ),
which provides a broader methodological overview. We recommend citing the
specific version of the library used in your research, along with the paper.
For citation examples, please refer to the
[documentation](https://tda-mapper.readthedocs.io/en/main/#citations).
