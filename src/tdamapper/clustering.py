"""
Clustering tools based on the Mapper algorithm.
"""

from tdamapper.core import mapper_connected_components, TrivialCover
import tdamapper.core
from tdamapper._common import ParamsMixin, clone, warn_deprecated


class TrivialClustering(tdamapper.core.TrivialClustering):
    """
    **DEPRECATED**: This class is deprecated and will be removed in a future
    release. Use :class:`tdamapper.core.TrivialClustering`.
    """
    def __init__(self):
        warn_deprecated(
            TrivialClustering.__qualname__,
            tdamapper.core.TrivialClustering.__qualname__,
        )
        super().__init__()


class FailSafeClustering(tdamapper.core.FailSafeClustering):
    """
    **DEPRECATED**: This class is deprecated and will be removed in a future
    release. Use :class:`tdamapper.core.FailSafeClustering`.
    """
    def __init__(self, clustering=None, verbose=True):
        warn_deprecated(
            FailSafeClustering.__qualname__,
            tdamapper.core.FailSafeClustering.__qualname__,
        )
        super().__init__(clustering, verbose)


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
    :param n_jobs: The maximum number of parallel clustering jobs. This
        parameter is passed to the constructor of :class:`joblib.Parallel`.
        Defaults to 1.
    :type n_jobs: int
    """

    def __init__(self, cover=None, clustering=None, n_jobs=1):
        self.cover = cover
        self.clustering = clustering
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        cover = TrivialCover() if self.cover is None \
            else self.cover
        cover = clone(cover)
        clustering = TrivialClustering() if self.clustering is None \
            else self.clustering
        clustering = clone(clustering)
        n_jobs = self.n_jobs
        y = X if y is None else y
        itm_lbls = mapper_connected_components(
            X,
            y,
            cover,
            clustering,
            n_jobs=n_jobs,
        )
        self.labels_ = [itm_lbls[i] for i, _ in enumerate(X)]
        return self
