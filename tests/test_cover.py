import unittest
import math
from tdamapper.cover import BallCover, KNNCover, CubicalCover


class TestCover(unittest.TestCase):

    def testBallCover(self):
        data = list(range(100))
        cover = BallCover(radius=10, metric=lambda x,y: abs(x - y))
        prox = cover.proximity()
        prox.fit(data)
        for x in data:
            result = prox.search(x)
            expected = [y for y in data if abs(x - y) <= 10]
            self.assertEqual(len(expected), len(result))

    def testKNNCover(self):
        data = list(range(100))
        cover = KNNCover(neighbors=11, metric=lambda x,y: abs(x - y))
        prox = cover.proximity()
        prox.fit(data)
        for x in range(5, 94):
            result = prox.search(x)
            expected = [x + i for i in range(-5, 6)]
            self.assertEqual(set(expected), set(result))

    def testCubicalCover(self):
        m, M = 0, 99
        n = 10
        p = 0.1
        w = (M - m) / (n * (1.0 - p))
        delta = p * w
        data = list(range(m, M + 1))
        cover = CubicalCover(n_intervals=n, overlap_frac=p)
        prox = cover.proximity()
        prox.fit(data)
        for x in data:
            result = prox.search(x)
            i = math.floor((x - m) / (w - delta))
            a_i = m + i * (w - delta) - delta / 2.0
            b_i = m + (i + 1) * (w - delta) + delta / 2.0
            expected = [y for y in data if y > a_i and y < b_i]
            self.assertEqual(set(expected), set(result))
