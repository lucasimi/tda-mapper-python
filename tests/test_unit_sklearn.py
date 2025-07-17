import logging

import numpy as np
import pytest
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.utils.estimator_checks import check_estimator

from tdamapper._common import clone
from tdamapper.core import TrivialClustering, TrivialCover
from tdamapper.cover import (
    BallCover,
    CubicalCover,
    KNNCover,
    ProximityCubicalCover,
    StandardCubicalCover,
)
from tdamapper.learn import MapperAlgorithm, MapperClustering
from tests.setup_logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _euclidean(x, y):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.linalg.norm(x - y)


def _test_clone(obj):
    obj_repr = repr(obj)
    obj_cln = clone(obj)
    cln_repr = repr(obj_cln)
    assert obj_repr == cln_repr


def _test_repr(obj):
    obj_repr = repr(obj)
    _obj = eval(obj_repr)
    _obj_repr = repr(_obj)
    assert obj_repr == _obj_repr


def _test_params(obj, params):
    obj.set_params(**params)
    params = obj.get_params(deep=True)
    for k, v in params.items():
        assert params[k] == v


@pytest.mark.parametrize(
    "clustering",
    [
        TrivialClustering(),
        # DBSCAN(eps=0.5, min_samples=5),
        # AgglomerativeClustering(n_clusters=3, linkage="ward"),
        # AgglomerativeClustering(n_clusters=3, linkage="average"),
    ],
)
@pytest.mark.parametrize(
    "cover",
    [
        TrivialCover(),
        BallCover(metric=_euclidean),
        BallCover(metric=_euclidean),
        KNNCover(metric=_euclidean),
        CubicalCover(n_intervals=3, overlap_frac=0.2),
        StandardCubicalCover(n_intervals=3, overlap_frac=0.2),
        ProximityCubicalCover(n_intervals=3, overlap_frac=0.2),
    ],
)
@pytest.mark.parametrize("estimator", [MapperAlgorithm, MapperClustering])
def test_mapper_estimator(estimator, cover, clustering):
    """
    Test that the estimator is compliant with scikit-learn's estimator checks.
    """
    est = estimator(cover=cover, clustering=clustering)
    for est_, check in check_estimator(est, generate_only=True):
        # logger.info(f'{check}')
        check(est_)


@pytest.mark.parametrize(
    "obj, params",
    [
        (TrivialClustering(), {}),
        (TrivialCover(), {}),
        (MapperAlgorithm(), {"cover": BallCover(metric=_euclidean)}),
        (
            MapperAlgorithm(),
            {
                "cover": BallCover(metric="euclidean"),
                "clustering": DBSCAN(eps=0.5, min_samples=5),
            },
        ),
        (MapperClustering(), {"cover": KNNCover(metric="euclidean")}),
        (BallCover(), {"radius": 0.5, "metric": "euclidean"}),
        (KNNCover(), {"neighbors": 5, "metric": "euclidean"}),
        (CubicalCover(), {"n_intervals": 3, "overlap_frac": 0.2}),
        (StandardCubicalCover(), {"n_intervals": 3, "overlap_frac": 0.2}),
        (ProximityCubicalCover(), {"n_intervals": 3, "overlap_frac": 0.2}),
    ],
)
def test_obj(obj, params):
    """
    Test that the object can be cloned, represented, and has correct parameters.
    """
    _test_clone(obj)
    _test_repr(obj)
    _test_params(obj, params)
