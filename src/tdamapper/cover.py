from tdamapper.proximity import BallProximity, KNNProximity, CubicalProximity, TrivialProximity


class ProximityNet:

    def __init__(self, X, proximity):
        self.__X = X
        self.__proximity = proximity

    def __iter__(self):
        '''
        Compute the proximity-net for a given open cover.

        :param X: A dataset
        :type X: numpy.ndarray or list-like
        :param cover: A cover algorithm
        :type cover: A class from tdamapper.cover
        '''
        covered_ids = set()
        self.__proximity.fit(self.__X)
        for i, xi in enumerate(self.__X):
            if i not in covered_ids:
                neigh_ids = self.__proximity.search(xi)
                covered_ids.update(neigh_ids)
                if neigh_ids:
                    yield neigh_ids


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

    def build(self, X):
        prox = BallProximity(self.radius, self.metric)
        return iter(ProximityNet(X, prox))


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

    def build(self, X):
        prox = KNNProximity(self.neighbors, self.metric)
        return iter(ProximityNet(X, prox))


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

    def build(self, X):
        prox = CubicalProximity(self.n_intervals, self.overlap_frac)
        return iter(ProximityNet(X, prox))


class TrivialCover:

    def build(self, X):
        prox = TrivialProximity()
        return iter(ProximityNet(X, prox))
