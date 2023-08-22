import unittest
import numpy as np

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import BallCover
from tdamapper.clustering import TrivialClustering
from tdamapper.plot import MapperPlot


def dist(x, y):
    return np.linalg.norm(x - y)


class TestMapperPlot(unittest.TestCase):

    def testTwoConnectedClusters(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])]
        mp = MapperAlgorithm(cover=BallCover(1.1, metric=dist),
            clustering=TrivialClustering())
        g = mp.fit_transform(data, data)
        mp_plot = MapperPlot(data, g)
        mp_plot.plot_static()
        mp_plot.plot_interactive_2d()
        mp_plot.plot_interactive_3d()
