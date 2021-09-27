import numpy as np

import mapper
import mapper.graph
from mapper.cover import SearchCover, TrivialCover
from mapper.clustering import ClusteringAlgorithm, TrivialClustering
from mapper.search import BallSearch, KnnSearch
from mapper.exact import Mapper
from mapper.plot import GraphPlot
from mapper.graph import GraphColormap


dist = lambda x, y: np.linalg.norm(x - y)
lens = lambda x: x

dim = 10
num = 100
data = [np.random.rand(dim) for _ in range(num)]
mp = Mapper(cover_algo=SearchCover(BallSearch(1.5)), clustering_algo=TrivialClustering())
g = mp.run(data, lens, dist)

gp = GraphPlot(g)

gc = GraphColormap(np.nanmean)
colors = gc.get_colors(g, data)

gp.plot(colors)
