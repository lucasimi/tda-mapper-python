from sklearn.cluster import DBSCAN

from tdamapper._common import clone
from tdamapper.cover import BallCover, CubicalCover
from tdamapper.learn import MapperAlgorithm, MapperClustering


def __test_clone(obj):
    obj_repr = repr(obj)
    obj_cln = clone(obj)
    cln_repr = repr(obj_cln)
    assert obj_repr == cln_repr


def __test_repr(obj):
    obj_repr = repr(obj)
    _obj = eval(obj_repr)
    _obj_repr = repr(_obj)
    assert obj_repr == _obj_repr


def __test_clone_and_repr(obj):
    __test_clone(obj)
    __test_repr(obj)


def test_params_mapper_algorithm():
    est = MapperAlgorithm(
        cover=CubicalCover(
            n_intervals=3,
            overlap_frac=0.3,
        ),
    )
    params = est.get_params(deep=False)
    assert 5 == len(params)
    params = est.get_params()
    assert 12 == len(params)
    assert 3 == params["cover__n_intervals"]
    assert 0.3 == params["cover__overlap_frac"]
    est.set_params(cover__n_intervals=2, cover__overlap_frac=0.2)
    params = est.get_params()
    assert 12 == len(params)
    assert 2 == params["cover__n_intervals"]
    assert 0.2 == params["cover__overlap_frac"]


def test_params_mapper_clustering():
    est = MapperClustering(
        cover=CubicalCover(
            n_intervals=3,
            overlap_frac=0.3,
        ),
    )
    params = est.get_params(deep=False)
    assert 3 == len(params)
    params = est.get_params()
    assert 10 == len(params)
    assert 3 == params["cover__n_intervals"]
    assert 0.3 == params["cover__overlap_frac"]
    est.set_params(cover__n_intervals=2, cover__overlap_frac=0.2)
    params = est.get_params()
    assert 10 == len(params)
    assert 2 == params["cover__n_intervals"]
    assert 0.2 == params["cover__overlap_frac"]


def test_clone_and_repr_ball_cover():
    __test_clone_and_repr(BallCover())
    __test_clone_and_repr(
        BallCover(
            radius=2.0,
            metric="test",
            metric_params={"f": 4},
            kind="kind_test",
            leaf_capacity=3.0,
            leaf_radius=-2.0,
            pivoting=7,
        )
    )


def test_clone_and_repr_cubical_cover():
    __test_clone_and_repr(CubicalCover())
    __test_clone_and_repr(
        CubicalCover(
            n_intervals=4,
            overlap_frac=5,
            algorithm="algo_test",
            kind="simple",
            leaf_radius=5,
            leaf_capacity=6,
            pivoting="no",
        )
    )


def test_clone_repr_mapper_algorithm():
    __test_clone_and_repr(MapperAlgorithm())
    __test_clone_and_repr(
        MapperAlgorithm(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
            clustering=DBSCAN(
                eps="none",
                min_samples=5.4,
            ),
            failsafe=4,
            n_jobs="foo",
            verbose=4,
        )
    )


def test_clone_repr_mapper_clustering():
    __test_clone_and_repr(MapperClustering())
    __test_clone_and_repr(
        MapperClustering(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
            clustering=DBSCAN(
                eps="none",
                min_samples=5.4,
            ),
            n_jobs="foo",
        )
    )
