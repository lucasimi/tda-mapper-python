from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT


class VPTree:
    """
    A Vantage Point Tree, or vp-tree, for fast range-queries and knn-queries.

    :param X: A dataset of n points to be covered with open subsets.
    :type X: array-like of shape (n, m) or list-like of length n

    :param metric: The metric used to define the distance between points.
    Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
    Defaults to 'euclidean'.
    :type metric: str or callable

    :param metric_params: Additional parameters for the metric function, to be
    passed to `tdamapper.utils.metrics.get_metric`. Defaults to None.
    :type metric_params: dict, optional

    :param kind: Specifies whether to use a flat or a hierarchical vantage
    point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults to
    'flat'.
    :type kind: str

    :param leaf_capacity: The maximum number of points in a leaf node of the
    vantage point tree. Must be a positive value. Defaults to 1.
    :type leaf_capacity: int

    :param leaf_radius: The radius of the leaf nodes. Must be a positive
    value. Defaults to 0.0.
    :type leaf_radius: float

    :param pivoting: The method used for pivoting in the vantage point tree.
    Acceptable values are None, 'random', or 'furthest'. Defaults to None.
    :type pivoting: str or callable, optional
    """

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
