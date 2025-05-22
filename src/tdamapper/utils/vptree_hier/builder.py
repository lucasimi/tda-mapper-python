from random import randrange

import numpy as np

from tdamapper.utils.quickselect import quickselect, swap_all
from tdamapper.utils.vptree_hier.common import Leaf, Node, VPArray, _mid


class Builder:

    def __init__(self, vpt, X):
        self.__distance = vpt._get_distance()

        dataset = [x for x in X]
        indices = np.array([i for i in range(len(dataset))])
        distances = np.array([0.0 for _ in X])
        self._arr = VPArray(dataset, distances, indices)

        self.__leaf_capacity = vpt.get_leaf_capacity()
        self.__leaf_radius = vpt.get_leaf_radius()
        pivoting = vpt.get_pivoting()
        self.__pivoting = self._pivoting_disabled
        if pivoting == "random":
            self.__pivoting = self._pivoting_random
        elif pivoting == "furthest":
            self.__pivoting = self._pivoting_furthest

    def _pivoting_disabled(self, start, end):
        pass

    def _pivoting_random(self, start, end):
        if end <= start:
            return
        pivot = randrange(start, end)
        if pivot > start:
            self._arr.swap(start, pivot)

    def _furthest(self, start, end, i):
        furthest_dist = 0.0
        furthest = start
        i_point = self._arr.get_point(i)
        for j in range(start, end):
            j_point = self._arr.get_point(j)
            j_dist = self.__distance(i_point, j_point)
            if j_dist > furthest_dist:
                furthest = j
                furthest_dist = j_dist
        return furthest

    def _pivoting_furthest(self, start, end):
        if end <= start:
            return
        rnd = randrange(start, end)
        furthest_rnd = self._furthest(start, end, rnd)
        furthest = self._furthest(start, end, furthest_rnd)
        if furthest > start:
            self._arr.swap(start, furthest)

    def _update(self, start, end):
        self.__pivoting(start, end)
        v_point = self._arr.get_point(start)
        for i in range(start + 1, end):
            point = self._arr.get_point(i)
            self._arr.set_distance(i, self.__distance(v_point, point))

    def build(self):
        tree = self._build_rec(0, self._arr.size())
        return tree, self._arr

    def _build_rec(self, start, end):
        mid = _mid(start, end)
        self._update(start, end)
        v_point = self._arr.get_point(start)
        self._arr.partition(start + 1, end, mid)
        v_radius = self._arr.get_distance(mid)
        self._arr.set_distance(start, v_radius)
        if (end - start <= 2 * self.__leaf_capacity) or (
            v_radius <= self.__leaf_radius
        ):
            left = Leaf(start + 1, mid)
            right = Leaf(mid, end)
        else:
            left = self._build_rec(start + 1, mid)
            right = self._build_rec(mid, end)
        return Node(v_radius, v_point, left, right)
