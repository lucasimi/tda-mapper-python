![Logo](https://github.com/lucasimi/tda-mapper-python/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png)

[![PyPI version](https://badge.fury.io/py/tda-mapper.svg)](https://badge.fury.io/py/tda-mapper)
[![downloads](https://img.shields.io/pypi/dm/tda-mapper)](https://pypi.python.org/pypi/tda-mapper/)
[![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml)
[![publish](https://github.com/lucasimi/tda-mapper-python/actions/workflows/publish.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/publish.yml)
[![docs](https://readthedocs.org/projects/tda-mapper/badge/?version=main)](https://tda-mapper.readthedocs.io/en/main/?badge=main)
[![codecov](https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R)](https://codecov.io/github/lucasimi/tda-mapper-python)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10642381.svg)](https://doi.org/10.5281/zenodo.10642381)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tda-mapper-app.streamlit.app/)

# tda-mapper

**tda-mapper** is a simple and efficient Python library implementing the Mapper algorithm for Topological Data Analysis (TDA).
It enables fast computation of Mapper graphs using *vp-trees* to optimize the construction of open covers for enhanced performance and scalability.

For further details, please refer to our [preprint](https://doi.org/10.5281/zenodo.10659651).

- **Installation**: `pip install tda-mapper`
- **Documentation**: [online on readthedocs](https://tda-mapper.readthedocs.io/en/main/)

## Features

- **Efficient Mapper Computation**: Optimized for higher-dimensional lenses.
- **Interactive Visualizations**: Multiple plotting backends for flexibility.
- **Data Exploration App**: Interactive tool for quick, in-depth data exploration.

## Quick Start

Here's a minimal example using the **circles dataset** from `scikit-learn` to demonstrate Mapper with **tda-mapper**:

```python
import numpy as np

from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

# load a labelled dataset
X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)
y = PCA(2).fit_transform(X)

cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
clust = DBSCAN()
graph = MapperAlgorithm(cover, clust).fit_transform(X, y)

# color according to labels
fig = MapperPlot(graph, dim=2, seed=42, iterations=60).plot_plotly(colors=labels)
fig.show(config={'scrollZoom': True})
```

| Original Dataset | Mapper Graph |
| ---------------- | ------------ |
| ![Original Dataset](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_dataset.png) | ![Mapper Graph](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_mean.png) |

More examples can be found in the
[documentation](https://tda-mapper.readthedocs.io/en/main/).

## Demo App

To assess the features of **tda-mapper** you can start from the demo app.

- **Live demo:** [tda-mapper-app on Streamlit Cloud](https://tda-mapper-app.streamlit.app/)

- **Run locally:** use the following commands

    ```
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py
    ```

## References and Citations

The Mapper algorithm is a well-known technique in the field of topological
data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining, and
social sciences, due to its ability to preserve topological features of the
underlying space, providing a visual representation that facilitates
exploration and interpretation. For an in-depth coverage of Mapper you can
read
[the original paper](https://research.math.osu.edu/tgda/mapperPBG.pdf).

- **tda-mapper**: To cite this library reference the Zenodo [archive](https://doi.org/10.5281/zenodo.10642381),
pointing to the specific version of the release used in your work.

- **Methodology**: To cite our methodological foundation, refer to [our preprint](https://doi.org/10.5281/zenodo.10659651).
