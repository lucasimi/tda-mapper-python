"""
Clustering tools based on the Mapper algorithm.
"""

from tdamapper.core import mapper_connected_components, TrivialCover
import tdamapper.core
from tdamapper._common import ParamsMixin, clone


class TrivialClustering(tdamapper.core.TrivialClustering):
    """
    Deprecated. Use :class:`tdamapper.core.TrivialClustering`.
    """
    pass


class FailSafeClustering(tdamapper.core.FailSafeClustering):
    """
    Deprecated. Use :class:`tdamapper.core.FailSafeClustering`.
    """
    pass


class MapperClustering(ParamsMixin):
    """
    A clustering algorithm based on the Mapper graph.

    The Mapper algorithm constructs a graph from a dataset, where each node
    represents a cluster of points and each edge represents an overlap between
    clusters. Each point in the dataset belongs to one or more nodes in the
    graph. These nodes are therefore all connected and share the same connected
    component in the Mapper graph. This class builds clusters of points
    according to their connected component in the Mapper graph.

    This class does not compute the Mapper graph but calls the function
    :func:`tdamapper.core.mapper_connected_components`, which is faster.

    :param cover: A cover algorithm.
    :type cover: A class compatible with :class:`tdamapper.core.Cover`
    :param clustering: The clustering algorithm to apply to each subset of the
        dataset.
    :type clustering: A class compatible with scikit-learn estimators from
        :mod:`sklearn.cluster`
    """

    def __init__(self, cover=None, clustering=None):
        self.cover = cover
        self.clustering = clustering

    def fit(self, X, y=None):
        cover = TrivialCover() if self.cover is None \
            else self.cover
        cover = clone(cover)
        clustering = TrivialClustering() if self.clustering is None \
            else self.clustering
        clustering = clone(clustering)
        y = X if y is None else y
        itm_lbls = mapper_connected_components(X, y, cover, clustering)
        self.labels_ = [itm_lbls[i] for i, _ in enumerate(X)]
        return self
