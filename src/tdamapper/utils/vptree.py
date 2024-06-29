from tdamapper.utils.vptree_flat import VPTree as FVPT
from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_sklearn import VPTree as SKVPT

class VPTree:

    def __init__(self,
            metric='euclidean',
            leaf_capacity=1,
            leaf_radius=0.0,
            strategy='random',
            kind='hierarchical'):
        self.metric = metric
        self.leaf_capacity = leaf_capacity
        self.leaf_radius = leaf_radius
        self.strategy = strategy
        self.kind = kind

    def fit(self, X):
        if self.kind == 'hierarchical':
            self.__vpt = HVPT(
                metric=self.metric,
                leaf_capacity=self.leaf_capacity, 
                leaf_radius=self.leaf_radius,
                strategy=self.strategy)
        elif self.kind == 'flat':
            self.__vpt = FVPT(
                metric=self.metric,
                leaf_capacity=self.leaf_capacity, 
                leaf_radius=self.leaf_radius,
                strategy=self.strategy)
        elif self.kind == 'sklearn':
            self.__vpt = SKVPT(
                metric=self.metric,
                leaf_capacity=self.leaf_capacity)
        self.__vpt.fit(X)
        return self

    def ball_search(self, point, eps=0.5, inclusive=True):
        return self.__vpt.ball_search(point=point, eps=eps, inclusive=inclusive)

    def knn_search(self, point, k=1):
        return self.__vpt.knn_search(point=point, k=k)
