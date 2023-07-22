import unittest

import networkx as nx

from mapper.cover import CoverGraph


class TestCover(unittest.TestCase):

    def testEmpty(self):
        lbls = CoverGraph().build_labels([])
        self.assertFalse(lbls)
        adj = CoverGraph().build_adjaciency([])
        self.assertFalse(adj)
        g = CoverGraph().build_nx(adj)
        self.assertFalse(g)

    def testSingleton(self):
        lbls = CoverGraph().build_labels([[1, 2, 3, 4]])
        adj = CoverGraph().build_adjaciency([[1, 2, 3, 4]])
        self.assertEqual(1, len(adj))
        self.assertEqual([], adj[0])
        g = CoverGraph().build_nx(adj)

    def testRepeated(self):
        lbls = CoverGraph().build_labels([[1, 2, 3, 4], [4, 3, 2, 1]])
        adj = CoverGraph().build_adjaciency([[1, 2, 3, 4], [4, 3, 2, 1]])
        self.assertEqual(2, len(adj))
        self.assertEqual([1], adj[0])
        self.assertEqual([0], adj[1])
        g = CoverGraph().build_nx(adj)
