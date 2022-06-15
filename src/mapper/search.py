from .utils.vptree import VPTree


class BallSearch:

    def __init__(self, radius, metric):
        self.__metric = metric
        self.__radius = radius
        self.__vptree = None

    def setup(self, data):
        self.__vptree = VPTree(self.__metric, data, leaf_radius=self.__radius)

    def find_neighbors(self, point):
        if self.__vptree:
            return self.__vptree.ball_search(point, self.__radius)
        else:
            return []


class KnnSearch:

    def __init__(self, k, metric):
        self.__k = k
        self.__metric = metric
        self.__vptree = None

    def setup(self, data):
        self.__vptree = VPTree(self.__metric, data, leaf_size=self.__k)

    def find_neighbors(self, point):
        if self.__vptree:
            return self.__vptree.knn_search(point, self.__k)
        else:
            return []


class TrivialSearch:

    def __init__(self):
        self.__data = None

    def setup(self, data):
        self.__data = data

    def fit_neighbors(self, x=None):
        return self.__data
