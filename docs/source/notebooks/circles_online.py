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
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA

from tdamapper.cover import CubicalCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

width, height, dpi = 500, 500, 100

# Generate toy dataset
X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)

fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=0.25, cmap="jet")
plt.axis("off")
plt.show()
# fig.savefig("circles_dataset.png", dpi=dpi)

# Apply PCA as lens
y = PCA(2, random_state=42).fit_transform(X)


# %% [markdown]
# ### Build Mapper graph

# %%
cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
clust = DBSCAN()
mapper = MapperAlgorithm(cover=cover, clustering=clust)
graph = mapper.fit_transform(X, y)

# %% [markdown]
# ### Plot Mapper graph with mean

# %%
plot = MapperPlot(graph, dim=2, iterations=60, seed=42)

fig = plot.plot_plotly(
    colors=labels,  # color according to categorical values
    cmap="jet",  # Jet colormap, for classes
    agg=np.nanmean,  # aggregate on nodes according to mean
    width=600,
    height=600,
)

fig.show(renderer="notebook_connected", config={"scrollZoom": True})
# fig.write_image("circles_mean.png", width=width, height=height)

# %%
plot.plot_plotly_update(
    fig,  # update the old figure
    colors=labels,
    cmap="viridis",  # viridis colormap, for ranges
    agg=np.nanstd,  # aggregate on nodes according to std
)

fig.show(renderer="notebook_connected", config={"scrollZoom": True})
# fig.write_image("circles_std.png", width=width, height=height)
