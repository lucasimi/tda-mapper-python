import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA

from tdamapper.cover import CubicalCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

# Generate toy dataset
X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, s=0.25, cmap="jet")
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
# fig.show(config={"scrollZoom": True})
