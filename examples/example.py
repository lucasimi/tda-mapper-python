import numpy as np

import mapper
import mapper.graph
from mapper.cover import SearchCover, TrivialCover
from mapper.clustering import ClusteringAlgorithm, TrivialClustering
from mapper.search import BallSearch, KnnSearch
from mapper.exact import Mapper
from mapper.plot import GraphPlot


dist = lambda x, y: np.linalg.norm(x - y)
lens = lambda x: x
data = [np.random.rand(10) for _ in range(1000)]
mp = Mapper(cover_algo=TrivialCover(), clustering_algo=TrivialClustering())
g = mp.run(data, lens, dist, np.nanmean)
gp = GraphPlot(g)
gp.plot()

