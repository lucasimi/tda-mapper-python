import numpy as np

import mapper
import mapper.graph
from mapper.cover import SearchCover, TrivialCover
from mapper.clustering import ClusteringAlgorithm, TrivialClustering
from mapper.search import BallSearch, KnnSearch
from mapper.exact import Mapper
from mapper.plot import GraphPlot

def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


lens = lambda x: x
data = dataset()
mp = Mapper(cover_algo=TrivialCover(), clustering_algo=TrivialClustering())
g = mp.run(data, lens, dist, np.nanmean)
gp = GraphPlot(g)
gp.plot()


