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
