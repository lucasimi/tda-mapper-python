import numpy as np

from sklearn.datasets import load_digits
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover
from tdamapper.clustering import PermissiveClustering
from tdamapper.plot import MapperPlot

# We load a labelled dataset
X, y = load_digits(return_X_y=True)             
# We compute the lens values
lens = PCA(2).fit_transform(X)                  

mapper_algo = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.65),
    # We prevent clustering failures
    clustering=PermissiveClustering(            
        clustering=AgglomerativeClustering(10),
        verbose=False),
    n_jobs=1)
mapper_graph = mapper_algo.fit_transform(X, lens)

mapper_plot = MapperPlot(X, mapper_graph,
    # We color according to digit values
    colors=y,                                   
    # Jet colormap, used for classes
    cmap='jet',                                 
    # We aggregate on graph nodes according to mean
    agg=np.nanmean,                             
    dim=2,
    iterations=400)
fig_mean = mapper_plot.plot(title='digit (mean)', width=600, height=600)
fig_mean.show(config={'scrollZoom': True})     

# We reuse the graph plot with the same positions
fig_std = mapper_plot.with_colors(              
    colors=y,
    # Viridis colormap, used for ranges
    cmap='viridis',                             
    # We aggregate on graph nodes according to std
    agg=np.nanstd,                              
).plot(title='digit (std)', width=600, height=600)
fig_std.show(config={'scrollZoom': True})      
