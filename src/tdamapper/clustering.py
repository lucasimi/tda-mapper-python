"""
Clustering tools based on the Mapper algorithm.
"""

from tdamapper.core import mapper_connected_components, TrivialCover
import tdamapper.core
from tdamapper._common import (
    ParamsMixin,
    EstimatorMixin,
    clone,
    warn_deprecated,
)


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


class _MapperClustering(EstimatorMixin, ParamsMixin):

    def __init__(self, cover=None, clustering=None, n_jobs=1):
        self.cover = cover
        self.clustering = clustering
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X, y = self._validate_X_y(X, y)
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
        self._set_n_features_in(X)
        return self


class MapperClustering(_MapperClustering):
    """
    **DEPRECATED**: This class is deprecated and will be removed in a future
    release. Use :class:`tdamapper.learn.MapperClustering`.
    """

    def __init__(self, cover=None, clustering=None, n_jobs=1):
        warn_deprecated(
            MapperClustering.__qualname__,
            'tdamapper.learn.MapperClustering',
        )
        super().__init__(
            cover=cover,
            clustering=clustering,
            n_jobs=n_jobs,
        )
