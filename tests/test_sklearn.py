import unittest
from sklearn.utils.estimator_checks import check_estimator

from tdamapper.clustering import CoverClustering, TrivialClustering


class TestSklearn(unittest.TestCase):

    def testEstimators(self):
        check_estimator(CoverClustering())
        check_estimator(TrivialClustering())
