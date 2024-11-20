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

**tda-mapper** is a simple and efficient Python library implementing the
Mapper algorithm for Topological Data Analysis (TDA).
It enables fast computation of Mapper graphs by using *vp-trees* to optimize
the construction of open covers, improving both performance and scalability.

For further details, please refer to the
[preprint](https://doi.org/10.5281/zenodo.10659651) and 
[online documentation](https://tda-mapper.readthedocs.io/en/main/).

- **Efficient Mapper Computation**: Optimized for higher-dimensional lenses.

- **Interactive Visualizations**: Multiple plotting backends for flexibility.

- **Interactive App**: Interactive tool for quick, in-depth data exploration.

### Background

The Mapper algorithm is a well-known technique in the field of topological
data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining, and
social sciences, due to its ability to preserve topological features of the
underlying space, providing a visual representation that facilitates
exploration and interpretation. For an in-depth coverage of Mapper you can
read
[the original paper](https://research.math.osu.edu/tgda/mapperPBG.pdf).


| Step 1 | Step 2 | Step 3 | Step 4 |
| ------ | ------ | ------ | ------ |
| ![Step 1](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_1.png) | ![Step 2](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_2.png) | ![Step 3](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_3.png) | ![Step 2](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_4.png) |
| Chose lens | Cover image | Run clustering | Build graph |

### Citations

If you use **tda-mapper** in your work, please consider citing both the
[library](https://doi.org/10.5281/zenodo.10642381), archived in a permanent
Zenodo record, and the [preprint](https://doi.org/10.5281/zenodo.10659651),
which provides a broader methodological overview.
We recommend citing the specific version of the library used in your research,
as well as the latest version of the preprint.
For citation examples, refer to the
[documentation](https://tda-mapper.readthedocs.io/en/main/#citations).

## Quick Start

### Installation

To install the latest version uploaded on PyPI

```bash
pip install tda-mapper
```

### How to Use

Here's a minimal example using the **circles dataset** from `scikit-learn` to
demonstrate how to use **tda-mapper**:

```python
import numpy as np

from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from tdamapper.learn import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

# load a labelled dataset
X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)
y = PCA(2, random_state=42).fit_transform(X)

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
[documentation](https://tda-mapper.readthedocs.io/en/main/examples.html).

### Interactive App

You can explore a live demo of **tda-mapper** directly on
[Streamlit Cloud](https://tda-mapper-app.streamlit.app/),
or run it locally using the following:

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```
