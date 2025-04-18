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
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from tdamapper.clustering import FailSafeClustering
from tdamapper.cover import CubicalCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

# We load a labelled dataset
X, labels = load_digits(return_X_y=True)

# Apply PCA as lens
y = PCA(2, random_state=42).fit_transform(X)

# %% [markdown]
# ### Build Mapper graph

# %%
algo = MapperAlgorithm(
    cover=CubicalCover(n_intervals=10, overlap_frac=0.5),
    clustering=AgglomerativeClustering(10),
    verbose=False,
)

graph = algo.fit_transform(X, y)

# %% [markdown]
# ### Plot Mapper graph with mean

# %%
plot = MapperPlot(graph, dim=3, iterations=400, seed=42)

fig = plot.plot_plotly(
    colors=labels,  # We color according to digit values
    cmap="jet",  # Jet colormap, used for classes
    agg=np.nanmean,  # We aggregate on graph nodes according to mean
    title="digit (mean)",
    width=600,
    height=600,
)

fig.show(renderer="notebook_connected", config={"scrollZoom": True})

# %% [markdown]
# ### Plot Mapper graph with standard deviation

# %%
fig = plot.plot_plotly(
    colors=labels,
    cmap="viridis",  # Viridis colormap, used for ranges
    agg=np.nanstd,  # We aggregate on graph nodes according to std
    title="digit (std)",
    width=600,
    height=600,
)

fig.show(renderer="notebook_connected", config={"scrollZoom": True})

# %% [markdown]
# ### Inspect interesting nodes

# %%
from matplotlib import pyplot as plt

# By interacting with the plot we see that node 140 is joining the cluster of
# digit 0 with the cluster of digit 4. Let's see how digits inside look like!

node_140 = [X[i, :] for i in graph.nodes()[140]["ids"]]
fig, axes = plt.subplots(1, len(node_140))
for dgt, ax in zip(node_140, axes):
    ax.imshow(dgt.reshape(8, 8), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()
