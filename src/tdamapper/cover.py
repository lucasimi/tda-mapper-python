from tdamapper.proximity import (
    BallProximity,
    KNNProximity,
    TrivialProximity,
    GridProximity,
    CubicalProximity)


class ProximityNetCover:

    def __init__(self, neighbors):
        self.__neighbors = neighbors

    def proximity_net(self, X):
        covered_ids = set()
        self.__neighbors.fit(X)
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh_ids = self.__neighbors.search(xi)
                covered_ids.update(neigh_ids)
                yield neigh_ids

    def get_params(self, deep=True):
        return self.__neighbors.get_params(deep)


class BallCover(ProximityNetCover):

    def __init__(self, radius, metric):
        super().__init__(BallProximity(radius, metric))


class KNNCover(ProximityNetCover):

    def __init__(self, k_neighbors, metric):
        super().__init__(KNNProximity(k_neighbors, metric))


class GridCover(ProximityNetCover):

    def __init__(self, n_intervals, overlap_frac):
        super().__init__(GridProximity(n_intervals, overlap_frac))


class CubicCover(ProximityNetCover):

    def __init__(self, n_intervals, overlap_frac):
        super().__init__(CubicalProximity(n_intervals, overlap_frac))


class TrivialCover(ProximityNetCover):

    def __init__(self):
        super().__init__(TrivialProximity())
