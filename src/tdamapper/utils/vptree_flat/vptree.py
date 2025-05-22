from tdamapper.utils.metrics import get_metric
from tdamapper.utils.vptree_flat.ball_search import BallSearch
from tdamapper.utils.vptree_flat.builder import Builder
from tdamapper.utils.vptree_flat.knn_search import KnnSearch


class VPTree:

    def __init__(
        self,
        X,
        metric="euclidean",
        metric_params=None,
        leaf_capacity=1,
        leaf_radius=0.0,
        pivoting=None,
    ):
        self.__metric = metric
        self.__metric_params = metric_params
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = pivoting
        self._arr = Builder(self, X).build()

    def get_metric(self):
        return self.__metric

    def get_metric_params(self):
        return self.__metric_params

    def get_leaf_capacity(self):
        return self.__leaf_capacity

    def get_leaf_radius(self):
        return self.__leaf_radius

    def get_pivoting(self):
        return self.__pivoting

    def _get_arr(self):
        return self._arr

    def _get_distance(self):
        metric_params = self.__metric_params or {}
        return get_metric(self.__metric, **metric_params)

    def ball_search(self, point, eps, inclusive=True):
        return BallSearch(self, point, eps, inclusive).search()

    def knn_search(self, point, k):
        return KnnSearch(self, point, k).search()
