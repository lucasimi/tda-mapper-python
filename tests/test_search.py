import unittest
import numpy as np

from mapper.search import BallSearch, KnnSearch


def dist(x, y):
    return np.linalg.norm(x - y)


def dataset(dim=10, num=1000):
    return [np.random.rand(dim) for _ in range(num)]


class TestMapper(unittest.TestCase):

    def testTrivial(self):
        data = dataset()
        bs = BallSearch(10, dist)
        bs.fit(data)
        for x in data:
            result = bs.neighbors(x)
            expected = [y for y in data if dist(x, y) <= 10]
            self.assertEqual(len(expected), len(result))


if __name__=='__main__':
    unittest.main()

