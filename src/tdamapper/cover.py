import numpy as np
from tdamapper.utils.vptree_flat import VPTree
from tdamapper.proximity import BallProximity, KNNProximity, CubicalProximity, TrivialProximity


class BallCover:
    '''
    Create an open cover made of overlapping open balls of fixed radius

    :param radius: The radius of open balls
    :type radius: float
    :param metric: The metric used to define open balls
    :type metric: function
    '''

    def __init__(self, radius, metric):
        self.metric = metric
        self.radius = radius

    def proximity(self):
        return BallProximity(self.radius, self.metric)


class KNNCover:
    '''
    Create an open cover where each open set containes a fixed number of neighbors, using KNN.

    :param neighbors: The number of neighbors
    :type neighbors: int
    :param metric: The metric used to search neighbors
    :type metric: function
    '''

    def __init__(self, neighbors, metric):
        self.neighbors = neighbors
        self.metric = metric

    def proximity(self):
        return KNNProximity(self.neighbors, self.metric)


class CubicalCover:
    '''
    Create an open cover of hypercubes of given 

    :param neighbors: The number of neighbors
    :type neighbors: int
    :param metric: The metric used to search neighbors
    :type metric: function
    '''

    def __init__(self, n_intervals, overlap_frac):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac

    def proximity(self):
        return CubicalProximity(self.n_intervals, self.overlap_frac)


class TrivialCover:
    
    def proximity(self):
        return TrivialProximity()

