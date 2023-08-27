from tdamapper.search import BallSearch, KnnSearch, TrivialSearch, CubicSearch


class NeighborsCover:

    def __init__(self, search):
        self.__search = search

    def compute_neighbors_net(self, X):
        covered_ids = set()
        self.__search.fit(X)
        for i, xi in enumerate(X):
            if i not in covered_ids:
                neigh_ids = self.__search.neighbors(xi)
                covered_ids.update(neigh_ids)
                yield neigh_ids

    def get_params(self, deep=True):
        return self.__search.get_params(deep)


class BallCover(NeighborsCover):

    def __init__(self, radius, metric):
        super().__init__(BallSearch(radius, metric))


class KnnCover(NeighborsCover):

    def __init__(self, neighbors, metric):
        super().__init__(KnnSearch(neighbors, metric))


class CubicCover(NeighborsCover):

    def __init__(self, n, perc):
        super().__init__(CubicSearch(n, perc))


class TrivialCover(NeighborsCover):

    def __init__(self):
        super().__init__(TrivialSearch())
