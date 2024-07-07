from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT


class VPTree:

    def __init__(
        self,
        X,
        metric='euclidean',
        metric_params=None,
        kind='flat',
        leaf_capacity=1,
        leaf_radius=0.0,
        pivoting=None
    ):
        builder = FVPT
        if kind == 'flat':
            builder = FVPT
        elif kind == 'hierarchical':
            builder = HVPT
        else:
            raise ValueError(f'Unknown kind of vptree: {kind}')
        self.__vpt = builder(
            X,
            metric=metric,
            metric_params=metric_params,
            leaf_capacity=leaf_capacity,
            leaf_radius=leaf_radius,
            pivoting=pivoting)

    def ball_search(self, point, eps, inclusive=True):
        return self.__vpt.ball_search(point, eps, inclusive=inclusive)

    def knn_search(self, point, k):
        return self.__vpt.knn_search(point, k)
