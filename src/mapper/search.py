from .utils.vptree import VPTree


class BallSearch:

    def __init__(self, radius):
        self.__radius = radius
        self.__vptree = None

    def setup(self, data, metric):
        self.__vptree = VPTree(metric, data, min_radius=self.__radius)

    def find_neighbors(self, point):
        if self.__vptree:
            return self.__vptree.ball_search(point, self.__radius)
        else:
            return []


class KnnSearch:

    def __init__(self, k):
        self.__k = k
        self.__vptree = None

    def setup(self, data, metric):
        self.__vptree = VPTree(metric, data, max_count=self.__k)

    def find_neighbors(self, point):
        if self.__vptree:
            return self.__vptree.knn_search(point, self.__k)
        else:
            return []

