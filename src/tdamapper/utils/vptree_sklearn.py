"""A class for fast knn and range searches, depending only on a given metric"""
from random import randrange

from tdamapper.utils.quickselect import quickselect_tuple
from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.metrics import get_metric

from sklearn.neighbors import BallTree


class VPTree:

    def __init__(self, 
            metric='euclidean',
            leaf_capacity=1,
            leaf_radius=0.0,
            strategy='random'):
        self.metric = metric
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius

    def fit(self, X):
        self.X = X
        self.ball_tree = BallTree(
            self.X,
            leaf_size=self.leaf_capacity,
            #metric=get_metric(self.metric),
            metric=self.metric
            )

    def ball_search(self, point, eps=0.5, inclusive=True):
        neighs = self.ball_tree.query_radius([point], r=eps)
        return [self.X[i] for i in neighs[0]]

    def knn_search(self, point, k=1):
        neighs = self.ball_tree.query([point], k=k, return_distance=False)
        return [self.X[i] for i in neighs[0]]
