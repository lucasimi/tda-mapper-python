import unittest

from tdamapper.utils.unionfind import UnionFind


class TestUnionFind(unittest.TestCase):

    def test_list(self):
        data = [1, 2, 3, 4]
        uf = UnionFind(data)
        for i in data:
            self.assertEqual(i, uf.find(i))
        j = uf.union(1, 2)
        self.assertEqual(j, uf.find(1))
        self.assertEqual(j, uf.find(2))
        k = uf.union(3, 4)
        self.assertEqual(k, uf.find(3))
        self.assertEqual(k, uf.find(4))
