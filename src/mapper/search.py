from .utils.balltree import BallTree


class BallSearch:

    def __init__(self, radius):
        self.__radius = radius
        self.__bt = None

    def setup(self, data, metric):
        self.__bt = BallTree(metric, data, min_radius=self.__radius)

    def find_neighbors(self, point):
        if self.__bt:
            return self.__bt.ball_search(point, self.__radius)
        else:
            return []


class KnnSearch:

    def __init__(self, k):
        self.__k = k
        self.__bt = None

    def setup(self, data, metric):
        self.__bt = BallTree(metric, data, max_count=self.__k)

    def find_neighbors(self, point):
        if self.__bt:
            return self.__bt.knn_search(point, self.__k)
        else:
            return []

