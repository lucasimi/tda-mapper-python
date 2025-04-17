# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Circles dataset

# %%
import numpy as np

from matplotlib import pyplot as plt

from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from tdamapper.learn import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot

X, y = make_circles(                # load a labelled dataset
    n_samples=5000,
    noise=0.05,
    factor=0.3,
    random_state=42
)
lens = PCA(2, random_state=42).fit_transform(X)

plt.scatter(lens[:, 0], lens[:, 1], c=y, cmap='jet')

# %% [markdown]
# ### Build Mapper graph

# %%
mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.3
    ),
    clustering=DBSCAN()
)

mapper_graph = mapper_algo.fit_transform(X, lens)

# %% [markdown]
# ### Plot Mapper graph with mean

# %%
mapper_plot = MapperPlot(
    mapper_graph,
    dim=2,
    iterations=60,
    seed=42
)

fig = mapper_plot.plot_plotly(
    colors=y,                       # color according to categorical values
    cmap='jet',                     # Jet colormap, for classes
    agg=np.nanmean,                 # aggregate on nodes according to mean
    width=600,
    height=600
)

fig.show(
    renderer='notebook_connected',
    config={'scrollZoom': True}
)

# %%
mapper_plot.plot_plotly_update(
    fig,                            # update the old figure
    colors=y,
    cmap='viridis',                 # viridis colormap, for ranges
    agg=np.nanstd                   # aggregate on nodes according to std
)                  
    
fig.show(
    renderer='notebook_connected',
    config={'scrollZoom': True}
)
