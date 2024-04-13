![Logo](https://github.com/lucasimi/tda-mapper-python/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png)

[![PyPI version](https://badge.fury.io/py/tda-mapper.svg)](https://badge.fury.io/py/tda-mapper)
[![downloads](https://img.shields.io/pypi/dm/tda-mapper)](https://pypi.python.org/pypi/tda-mapper/)
[![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml)
[![deploy](https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml)
[![docs](https://readthedocs.org/projects/tda-mapper/badge/?version=main)](https://tda-mapper.readthedocs.io/en/main/?badge=main)
[![codecov](https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R)](https://codecov.io/github/lucasimi/tda-mapper-python)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10659652.svg)](https://doi.org/10.5281/zenodo.10659652)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10642381.svg)](https://doi.org/10.5281/zenodo.10642381)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tda-mapper-app.streamlit.app/)

# tda-mapper

A simple and efficient Python implementation of Mapper algorithm for Topological Data Analysis

* **Installation**: `pip install tda-mapper`

* **Documentation**: https://tda-mapper.readthedocs.io/en/main/

* **Demo App**: https://tda-mapper-app.streamlit.app/

The Mapper algorithm is a well-known technique in the field of topological data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining, and social sciences, due to its ability to preserve topological features of the underlying space, providing a visual representation that facilitates exploration and interpretation.
For an in-depth coverage of Mapper you can read [the original paper](https://research.math.osu.edu/tgda/mapperPBG.pdf).

| Step 1 | Step 2 | Step 3 | Step 4 |
| ------ | ------ | ------ | ------ |
| ![Step 1](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_1.png) | ![Step 2](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_2.png) | ![Step 3](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_3.png) | ![Step 2](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_4.png) |
| Chose lens | Cover image | Run clustering | Build graph |

## Example

[Here](https://github.com/lucasimi/tda-mapper-python/raw/main/tests/example.py) you can find an example to use to kickstart your analysis.
In this toy-example we use a two-dimensional dataset of two concentric circles.
The Mapper graph is a topological summary of the whole point cloud.

```python
import numpy as np

from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperLayoutInteractive

X, y = make_circles(                # load a labelled dataset
    n_samples=5000,
    noise=0.05,
    factor=0.3,
    random_state=42)
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.3),
    clustering=DBSCAN())
mapper_graph = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperLayoutInteractive(
    mapper_graph,
    colors=y,                       # color according to categorical values
    cmap='jet',                     # Jet colormap, for classes
    agg=np.nanmean,                 # aggregate on nodes according to mean
    dim=2,
    iterations=60,
    seed=42,
    width=600,
    height=600)

fig_mean = mapper_plot.plot()
fig_mean.show(config={'scrollZoom': True})

mapper_plot.update(                 # reuse the plot with the same positions
    colors=y,
    cmap='viridis',                 # viridis colormap, for ranges
    agg=np.nanstd,                  # aggregate on nodes according to std
)

fig_std = mapper_plot.plot()
fig_std.show(config={'scrollZoom': True})
```

| Dataset | Mapper graph (average) | Mapper graph (deviation) |
| ------- | ---------------------- | ------------------------ |
| ![Dataset](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_dataset.png) | ![Mapper graph (average)](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_mean.png) | ![Mapper graph (standard deviation)](https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_std.png) |

More examples can be found in the documentation https://tda-mapper.readthedocs.io/en/main/.

### Demo App

You can also run a demo app locally by running

```
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Citations

If you want to use **tda-mapper** in your research please use one of the following citation.
For the methodology you can use:

```
Simi, L. (2024). A Scalable Implementation of Mapper for Topological Data Analysis via Vantage Point Trees. Zenodo. https://doi.org/10.5281/zenodo.10659652
```

BibTeX entry:


```
@misc{simi_2024_10659652,
    author       = {Simi, Luca},
    title        = {A Scalable Implementation of Mapper for Topological Data Analysis via Vantage Point Trees},
    month        = feb,
    year         = 2024,
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.10659652},
    url          = {https://doi.org/10.5281/zenodo.10659652}
}
```

If you want to refer to the actual library instead, you can reference the Zenodo 
archive [https://doi.org/10.5281/zenodo.10642381](https://doi.org/10.5281/zenodo.10642381>).
In the archive you can find a permanent reference to the exact version used in your work. 
