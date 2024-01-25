import numpy as np

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.plot import MapperPlot


X, y = load_digits(return_X_y=True)
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(n_intervals=10, overlap_frac=0.65),
    clustering=AgglomerativeClustering(10),
    n_jobs=1)
mapper_graph = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(X, mapper_graph,
    colors=y, 
    cmap='jet', 
    agg=np.nanmean,
    dim=2,
    iterations=400)
fig_mean = mapper_plot.plot(title='digit (mean)', width=600, height=600)
fig_mean.show(config={'scrollZoom': True})

fig_std = mapper_plot.with_colors(
    colors=y, 
    cmap='viridis', 
    agg=np.nanstd,
).plot(title='digit (std)', width=600, height=600)
fig_std.show(config={'scrollZoom': True})