import unittest

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import (
    BallCover,
    KNNCover,
    CubicalCover
)
from tdamapper.clustering import MapperClustering


class TestParams(unittest.TestCase):

    def test_params_mapper(self):
        est = MapperAlgorithm(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
        )
        params = est.get_params(deep=False)
        self.assertEquals(5, len(params))
        params = est.get_params()
        self.assertEquals(12, len(params))
        self.assertEquals(3, params['cover__n_intervals'])
        self.assertEquals(0.3, params['cover__overlap_frac'])
        est.set_params(cover__n_intervals=2, cover__overlap_frac=0.2)
        params = est.get_params()
        self.assertEquals(12, len(params))
        self.assertEquals(2, params['cover__n_intervals'])
        self.assertEquals(0.2, params['cover__overlap_frac'])

    def test_params_clust(self):
        est = MapperClustering(
            cover=CubicalCover(
                n_intervals=3,
                overlap_frac=0.3,
            ),
        )
        params = est.get_params(deep=False)
        self.assertEquals(3, len(params))
        params = est.get_params()
        self.assertEquals(10, len(params))
        self.assertEquals(3, params['cover__n_intervals'])
        self.assertEquals(0.3, params['cover__overlap_frac'])
        est.set_params(cover__n_intervals=2, cover__overlap_frac=0.2)
        params = est.get_params()
        self.assertEquals(10, len(params))
        self.assertEquals(2, params['cover__n_intervals'])
        self.assertEquals(0.2, params['cover__overlap_frac'])