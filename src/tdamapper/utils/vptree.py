"""
A module for fast knn and range searches, depending only on a given metric
"""

from tdamapper.utils.vptree_hier import VPTree as HVPT
from tdamapper.utils.vptree_flat import VPTree as FVPT


class VPTree:
    """
    A Vantage Point Tree, or vp-tree, for fast range-queries and knn-queries.

    :param X: A dataset of n points.
    :type X: array-like of shape (n, m) or list-like of length n
    :param metric: The metric used to define the distance between points.
        Accepts any value compatible with `tdamapper.utils.metrics.get_metric`.
        Defaults to 'euclidean'.
    :type metric: str or callable
    :param metric_params: Additional parameters for the metric function, to be
        passed to `tdamapper.utils.metrics.get_metric`. Defaults to None.
    :type metric_params: dict, optional
    :param kind: Specifies whether to use a flat or a hierarchical vantage
        point tree. Acceptable values are 'flat' or 'hierarchical'. Defaults
        to 'flat'.
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
            pivoting=pivoting
        )

    def ball_search(self, point, eps, inclusive=True):
        """
        Perform a ball search in the Vantage Point Tree.

        This method searches for all points within a specified radius from a
        given point.

        :param point: The query point from which to search for neighbors.
        :type point: objet, list, or array-like
        :param eps: The radius within which to search for neighbors. Must be
            positive.
        :type eps: float
        :param inclusive: Whether to include points exactly at the distance
            `eps` from `point`. Defaults to True.
        :type inclusive: bool
        :return: A list of points within the specified radius from the given
            query point.
        :rtype: list
        """
        return self.__vpt.ball_search(point, eps, inclusive=inclusive)

    def knn_search(self, point, k):
        """
        Perform a k-nearest neighbors search in the Vantage Point Tree.

        This method searches for the k-nearest neighbors to a given query
        point.

        :param point: The point from which to search for nearest neighbors.
        :type point: objet, list, or array-like
        :param k: The number of nearest neighbors to search for. Must be
            positive.
        :type k: int
        :return: A list of the k-nearest neighbors to the given query point.
        :rtype: list
        """
        return self.__vpt.knn_search(point, k)
