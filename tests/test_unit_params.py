import unittest

from sklearn.cluster import DBSCAN

from tdamapper._common import clone
from tdamapper.core import MapperAlgorithm
from tdamapper.cover import (
    BallCover,
    KNNCover,
    CubicalCover
)
from tdamapper.clustering import MapperClustering


class TestParams(unittest.TestCase):

    def __test_clone(self, obj):
        obj_repr = repr(obj)
        obj_cln = clone(obj)
        cln_repr = repr(obj_cln)
        self.assertEqual(obj_repr, cln_repr)

    def __test_repr(self, obj):
        obj_repr = repr(obj)
        _obj = eval(obj_repr)
        _obj_repr = repr(_obj)
        self.assertEqual(obj_repr, _obj_repr)

    def __test_clone_and_repr(self, obj):
        self.__test_clone(obj)
        self.__test_repr(obj)

    def test_params_mapper_algorithm(self):
        est = MapperAlgorithm(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
        )
        params = est.get_params(deep=False)
        self.assertEqual(5, len(params))
        params = est.get_params()
        self.assertEqual(12, len(params))
        self.assertEqual(3, params['cover__n_intervals'])
        self.assertEqual(0.3, params['cover__overlap_frac'])
        est.set_params(cover__n_intervals=2, cover__overlap_frac=0.2)
        params = est.get_params()
        self.assertEqual(12, len(params))
        self.assertEqual(2, params['cover__n_intervals'])
        self.assertEqual(0.2, params['cover__overlap_frac'])

    def test_params_mapper_clustering(self):
        est = MapperClustering(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
        )
        params = est.get_params(deep=False)
        self.assertEqual(3, len(params))
        params = est.get_params()
        self.assertEqual(10, len(params))
        self.assertEqual(3, params['cover__n_intervals'])
        self.assertEqual(0.3, params['cover__overlap_frac'])
        est.set_params(cover__n_intervals=2, cover__overlap_frac=0.2)
        params = est.get_params()
        self.assertEqual(10, len(params))
        self.assertEqual(2, params['cover__n_intervals'])
        self.assertEqual(0.2, params['cover__overlap_frac'])

    def test_clone_and_repr_ball_cover(self):
        self.__test_clone_and_repr(BallCover())
        self.__test_clone_and_repr(BallCover(
            radius=2.0,
            metric='test',
            metric_params={'f': 4},
            kind='kind_test',
            leaf_capacity=3.0,
            leaf_radius=-2.0,
            pivoting=7,
        ))

    def test_clone_and_repr_cubical_cover(self):
        self.__test_clone_and_repr(CubicalCover())
        self.__test_clone_and_repr(CubicalCover(
            n_intervals=4,
            overlap_frac=5,
            algorithm='algo_test',
            kind='simple',
            leaf_radius=5,
            leaf_capacity=6,
            pivoting='no'
        ))

    def test_clone_repr_mapper_algorithm(self):
        self.__test_clone_and_repr(MapperAlgorithm())
        self.__test_clone_and_repr(MapperAlgorithm(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
            clustering=DBSCAN(
                eps='none',
                min_samples=5.4,
            ),
            failsafe=4,
            n_jobs='foo',
            verbose=4,
        ))

    def test_clone_repr_mapper_clustering(self):
        self.__test_clone_and_repr(MapperClustering())
        self.__test_clone_and_repr(MapperClustering(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
            clustering=DBSCAN(
                eps='none',
                min_samples=5.4,
            ),
            n_jobs='foo',
        ))
