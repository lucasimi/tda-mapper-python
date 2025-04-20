import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA

from tdamapper.cover import CubicalCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

X, y = make_circles(  # load a labelled dataset
    n_samples=5000, noise=0.05, factor=0.3, random_state=42
)
lens = PCA(2).fit_transform(X)

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(n_intervals=10, overlap_frac=0.3), clustering=DBSCAN()
)
mapper_graph = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(mapper_graph, dim=2, iterations=60, seed=42)

fig = mapper_plot.plot_plotly(
    title="",
    width=600,
    height=600,
    colors=y,  # color according to categorical values
    cmap="jet",  # Jet colormap, for classes
    agg=np.nanmean,  # aggregate on nodes according to mean
)

fig.show(config={"scrollZoom": True})

mapper_plot.plot_plotly_update(
    fig,  # reuse the plot with the same positions
    colors=y,
    cmap="viridis",  # viridis colormap, for ranges
    agg=np.nanstd,  # aggregate on nodes according to std
)

fig.show(config={"scrollZoom": True})
