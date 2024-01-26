import numpy as np

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.clustering import PermissiveClustering
from tdamapper.plot import MapperPlot

X, y = load_digits(return_X_y=True)             # We load a labelled dataset
lens = PCA(2).fit_transform(X)                  # We compute the lens values

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.65),
    clustering=PermissiveClustering(            # We prevent clustering failures
        clustering=AgglomerativeClustering(10),
        verbose=False),
    n_jobs=1)
mapper_graph = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(X, mapper_graph,
    colors=y,                                   # We color according to digit values
    cmap='jet',                                 # Jet colormap, used for classes
    agg=np.nanmean,                             # We aggregate on graph nodes according to mean
    dim=2,
    iterations=400)
fig_mean = mapper_plot.plot(title='digit (mean)', width=600, height=600)
#fig_mean.show(config={'scrollZoom': True})     # Uncomment to show the plot

fig_std = mapper_plot.with_colors(              # We reuse the graph plot with the same positions
    colors=y,
    cmap='viridis',                             # Virtidis colormap, used for ranges
    agg=np.nanstd,                              # We aggregate on graph nodes according to std
).plot(title='digit (std)', width=600, height=600)
#fig_std.show(config={'scrollZoom': True})      # Uncomment to show the plot
