from tdamapper.neighbors import BallNeighbors, KNNeighbors, TrivialNeighbors, GridNeighbors


class NeighborsCover:

    def __init__(self, neighbors):
        self.__neighbors = neighbors

    def neighbors_net(self, X):
        covered_ids = set()
        self.__neighbors.fit(X)
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh_ids = self.__neighbors.search(xi)
                covered_ids.update(neigh_ids)
                yield neigh_ids

    def get_params(self, deep=True):
        return self.__neighbors.get_params(deep)


class BallCover(NeighborsCover):

    def __init__(self, radius, metric):
        super().__init__(BallNeighbors(radius, metric))


class KNNCover(NeighborsCover):

    def __init__(self, k_neighbors, metric):
        super().__init__(KNNeighbors(k_neighbors, metric))


class GridCover(NeighborsCover):

    def __init__(self, n_intervals, overlap_frac):
        super().__init__(GridNeighbors(n_intervals, overlap_frac))


class TrivialCover(NeighborsCover):

    def __init__(self):
        super().__init__(TrivialNeighbors())







