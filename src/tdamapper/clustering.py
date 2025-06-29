"""
Clustering tools based on the Mapper algorithm.
"""

from __future__ import annotations

from typing import List, Optional

import tdamapper.core
from tdamapper._common import EstimatorMixin, ParamsMixin, clone, deprecated
from tdamapper.core import (
    ArrayLike,
    Clustering,
    Cover,
    TrivialCover,
    mapper_connected_components,
)


class TrivialClustering(tdamapper.core.TrivialClustering):
    """
    **DEPRECATED**: This class is deprecated and will be removed in a future
    release. Use :class:`tdamapper.core.TrivialClustering`.
    """

    @deprecated(
        "This class is deprecated and will be removed in a future release. "
        "Use tdamapper.core.TrivialClustering."
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FailSafeClustering(tdamapper.core.FailSafeClustering):
    """
    **DEPRECATED**: This class is deprecated and will be removed in a future
    release. Use :class:`tdamapper.core.FailSafeClustering`.
    """

    @deprecated(
        "This class is deprecated and will be removed in a future release. "
        "Use tdamapper.core.FailSafeClustering."
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _MapperClustering(EstimatorMixin, ParamsMixin):
    """
    Mapper clustering model that fits the Mapper algorithm to the data.

    This class is designed to be used with the Mapper algorithm for clustering
    data points based on their features. It allows for customization of the
    cover and clustering methods used in the Mapper algorithm.

    :param cover: The cover method to use for the Mapper algorithm. If None,
        a trivial cover will be used.
    :param clustering: The clustering method to use for the Mapper algorithm.
        If None, a trivial clustering will be used.
    :param n_jobs: The number of jobs to run in parallel. Default is 1.
    """

    labels_: List[int]

    def __init__(
        self,
        cover: Optional[Cover] = None,
        clustering: Optional[Clustering] = None,
        n_jobs: int = 1,
    ):
        self.cover = cover
        self.clustering = clustering
        self.n_jobs = n_jobs

    def fit(
        self, x_arr: ArrayLike, y_arr: Optional[ArrayLike] = None
    ) -> _MapperClustering:
        """
        Fit the Mapper clustering model to the data.

        :param x_arr: The input features array.
        :param y_arr: The target values array. If None, `x_arr` is used as `y_arr`.
        :return: The fitted Mapper clustering model.
        """
        y_arr = x_arr if y_arr is None else y_arr
        x_arr, y_arr = self._validate_x_y(x_arr, y_arr)
        cover = TrivialCover() if self.cover is None else self.cover
        cover = clone(cover)
        clustering = (
            tdamapper.core.TrivialClustering()
            if self.clustering is None
            else self.clustering
        )
        clustering = clone(clustering)
        n_jobs = self.n_jobs
        itm_lbls = mapper_connected_components(
            x_arr,
            y_arr,
            cover,
            clustering,
            n_jobs=n_jobs,
        )
        self.labels_ = [itm_lbls[i] for i, _ in enumerate(x_arr)]
        self._set_n_features_in(x_arr)
        return self


class MapperClustering(_MapperClustering):
    """
    **DEPRECATED**: This class is deprecated and will be removed in a future
    release. Use :class:`tdamapper.learn.MapperClustering`.
    """

    @deprecated(
        "This class is deprecated and will be removed in a future release. "
        "Use tdamapper.learn.MapperClustering."
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
