# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Digits dataset

# %%
import numpy as np

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.learn import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.clustering import FailSafeClustering
from tdamapper.plot import MapperPlot


X, y = load_digits(return_X_y=True)                 # We load a labelled dataset
lens = PCA(2, random_state=42).fit_transform(X)     # We compute the lens values

# %% [markdown]
# ### Build Mapper graph

# %%
mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.65
    ),
    clustering=AgglomerativeClustering(10),
    verbose=False
)

mapper_graph = mapper_algo.fit_transform(X, lens)

# %% [markdown]
# ### Plot Mapper graph with mean

# %%
mapper_plot = MapperPlot(
    mapper_graph,
    dim=2,
    iterations=400,
    seed=42
)

fig = mapper_plot.plot_plotly(
    colors=y,                        # We color according to digit values
    cmap='jet',                      # Jet colormap, used for classes
    agg=np.nanmean,                  # We aggregate on graph nodes according to mean
    title='digit (mean)',
    width=600,
    height=600
)

fig.show(
    renderer='notebook_connected',
    config={'scrollZoom': True}
)

# %% [markdown]
# ### Plot Mapper graph with standard deviation

# %%
fig = mapper_plot.plot_plotly(
    colors=y,                        
    cmap='viridis',                  # Viridis colormap, used for ranges
    agg=np.nanstd,                   # We aggregate on graph nodes according to std
    title='digit (std)',
    width=600,
    height=600
)

fig.show(
    renderer='notebook_connected',
    config={'scrollZoom': True}
)
