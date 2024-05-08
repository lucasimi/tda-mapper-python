import unittest

import numpy as np
from tdamapper.proximity import CubicalProximity, proximity_net


class TestLandmarks(unittest.TestCase):

    def testExpandSimple(self):
        X = [0.0, 2.0, 1.45, 1.55]
        prox = CubicalProximity(
            n_intervals=2,
            overlap_frac=0.5)
        pn = list(proximity_net(X, prox))
        self.assertEqual(3, len(pn))

    def testExpand(self):
        X = np.array([[0.0], [2.0], [1.45], [1.55]])
        prox = CubicalProximity(
            n_intervals=2,
            overlap_frac=0.5)
        pn = list(proximity_net(X, prox))
        self.assertEqual(3, len(pn))
        