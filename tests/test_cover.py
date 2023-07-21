import unittest

import networkx as nx

from mapper.cover import CoverGraph


class TestCover(unittest.TestCase):

    def testEmpty(self):
        adj = CoverGraph().build_dict([])
        g = CoverGraph().build_nx([])
        self.assertFalse(adj)
        self.assertFalse(g)

    def testSingleton(self):
        adj = CoverGraph().build_dict([[1, 2, 3, 4]])
        g = CoverGraph().build_nx([[1, 2, 3, 4]])
        self.assertEqual(1, len(adj))
        self.assertEqual(set(), adj[0])

    def testRepeated(self):
        adj = CoverGraph().build_dict([[1, 2, 3, 4], [4, 3, 2, 1]])
        g = CoverGraph().build_nx([[1, 2, 3, 4], [4, 3, 2, 1]])
        self.assertEqual(2, len(adj))
        self.assertEqual({1}, adj[0])
        self.assertEqual({0}, adj[1])
