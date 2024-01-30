from tdamapper.proximity import (
    proximity_net,
    BallProximity,
    KNNProximity,
    CubicalProximity,
    TrivialProximity)


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

    def apply(self, X):
        proximity = BallProximity(self.radius, self.metric)
        return proximity_net(X, proximity)


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

    def apply(self, X):
        proximity = KNNProximity(self.neighbors, self.metric)
        return proximity_net(X, proximity)


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

    def apply(self, X):
        proximity = CubicalProximity(self.n_intervals, self.overlap_frac)
        return proximity_net(X, proximity)


class TrivialCover:
    '''
    Create an open cover made of a single open set that contains the whole dataset
    '''

    def apply(self, X):
        proximity = TrivialProximity()
        return proximity_net(X, proximity)
