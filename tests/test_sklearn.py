import unittest
from sklearn.utils.estimator_checks import check_estimator

from tdamapper.estimator import MapperEstimator


class TestSklearn(unittest.TestCase):

    def testMapper(self):
        mapper_est = MapperEstimator()
        check_estimator(mapper_est)
