from .utils.vptree import VPTree


class BallSearch:

    def __init__(self, radius, metric):
        self.__metric = lambda x, y: metric(x[1], y[1])
        self.__radius = radius
        self.__vptree = None
        self.__data = None

    def fit(self, data):
        self.__data = list(enumerate(data))
        self.__vptree = VPTree(
            self.__metric, self.__data, leaf_radius=self.__radius)
        return self

    def neighbors(self, point):
        if self.__vptree:
            neighs = self.__vptree.ball_search((-1, point), self.__radius)
            return [x for (x, _) in neighs]
        else:
            return []


class KnnSearch:

    def __init__(self, k, metric):
        self.__k = k
        self.__metric = lambda x, y: metric(x[1], y[1])
        self.__vptree = None
        self.__data = None

    def fit(self, data):
        self.__data = list(enumerate(data))
        self.__vptree = VPTree(self.__metric, self.__data, leaf_size=self.__k)
        return self

    def neighbors(self, point):
        if self.__vptree:
            neighs = self.__vptree.knn_search((-1, point), self.__k)
            return [x for (x, _) in neighs]
        else:
            return []


class TrivialSearch:

    def __init__(self):
        self.__data = None

    def fit(self, data):
        self.__data = data
        return self

    def neighbors(self, point=None):
        return list(range(len(self.__data)))
