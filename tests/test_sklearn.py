import unittest
from sklearn.utils.estimator_checks import check_estimator

from mapper.clustering import CoverClustering, TrivialClustering
from mapper.cover import TrivialCover


class TestSklearn(unittest.TestCase):

    def testEstimators(self):
        check_estimator(CoverClustering())
        check_estimator(TrivialClustering())
