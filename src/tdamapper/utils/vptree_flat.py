from random import randrange

import numpy as np

from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.metrics import get_metric
from tdamapper.utils.quickselect import quickselect, swap_all


def _mid(start, end):
    return (start + end) // 2


class VPTree:

    def __init__(
        self,
        X,
        metric="euclidean",
        metric_params=None,
        leaf_capacity=1,
        leaf_radius=0.0,
        pivoting=None,
    ):
        self.__metric = metric
        self.__metric_params = metric_params
        self.__leaf_capacity = leaf_capacity
        self.__leaf_radius = leaf_radius
        self.__pivoting = pivoting
        (
            self.__dataset,
            self.__arr_distances,
            self.__arr_indices,
            self.__arr_is_terminal,
        ) = self._Build(self, X).build()

    def get_metric(self):
        return self.__metric

    def get_metric_params(self):
        return self.__metric_params

    def get_leaf_capacity(self):
        return self.__leaf_capacity

    def get_leaf_radius(self):
        return self.__leaf_radius

    def get_pivoting(self):
        return self.__pivoting

    def _get_dataset(self):
        return self.__dataset

    def _get_distance(self):
        metric_params = self.__metric_params or {}
        return get_metric(self.__metric, **metric_params)

    def _get_distances(self):
        return self.__arr_distances

    def _get_indices(self):
        return self.__arr_indices

    def _get_is_terminal(self):
        return self.__arr_is_terminal

    class _Build:

        def __init__(self, vpt, X):
            self.__distance = vpt._get_distance()

            self.__dataset = [x for x in X]
            self.__arr_indices = np.array([i for i in range(len(self.__dataset))])
            self.__arr_distances = np.array([0.0 for _ in X])
            self.__arr_is_terminal = np.array([False for _ in X])

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
                swap_all(
                    self.__arr_distances,
                    start,
                    pivot,
                    self.__arr_indices,
                    self.__arr_is_terminal,
                )

        def _get_point(self, i):
            return self.__dataset[self.__arr_indices[i]]

        def _furthest(self, start, end, i):
            furthest_dist = 0.0
            furthest = start

            i_point = self._get_point(i)

            for j in range(start, end):
                j_point = self._get_point(j)

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
                swap_all(
                    self.__arr_distances,
                    start,
                    furthest,
                    self.__arr_indices,
                    self.__arr_is_terminal,
                )

        def _update(self, start, end):
            self.__pivoting(start, end)

            v_point_index = self.__arr_indices[start]
            v_point = self.__dataset[v_point_index]
            is_terminal = self.__arr_is_terminal[start]

            for i in range(start + 1, end):
                point_index = self.__arr_indices[i]
                point = self.__dataset[point_index]

                self.__arr_distances[i] = self.__distance(v_point, point)
                self.__arr_is_terminal[i] = is_terminal

        def build(self):
            self._build_iter()
            return (
                self.__dataset,
                self.__arr_distances,
                self.__arr_indices,
                self.__arr_is_terminal,
            )

        def _build_iter(self):
            stack = [(0, len(self.__dataset))]
            while stack:
                start, end = stack.pop()
                mid = _mid(start, end)
                self._update(start, end)

                # v_point_index = self.__indices[start]

                quickselect(
                    self.__arr_distances,
                    start + 1,
                    end,
                    mid,
                    self.__arr_indices,
                    self.__arr_is_terminal,
                )

                v_radius = self.__arr_distances[mid]

                if (end - start > 2 * self.__leaf_capacity) and (
                    v_radius > self.__leaf_radius
                ):
                    self.__arr_distances[start] = v_radius
                    # self.__indices[start] = v_point_index
                    self.__arr_is_terminal[start] = False

                    stack.append((mid, end))
                    stack.append((start + 1, mid))
                else:
                    self.__arr_distances[start] = v_radius
                    # self.__indices[start] = v_point_index
                    self.__arr_is_terminal[start] = True

    def ball_search(self, point, eps, inclusive=True):
        return self._BallSearch(self, point, eps, inclusive).search()

    class _BallSearch:

        def __init__(self, vpt, point, eps, inclusive=True):
            self.__dataset = vpt._get_dataset()
            self.__distances = vpt._get_distances()
            self.__indices = vpt._get_indices()
            self.__is_terminal = vpt._get_is_terminal()
            self.__distance = vpt._get_distance()
            self.__point = point
            self.__eps = eps
            self.__inclusive = inclusive

        def search(self):
            return self._search_iter()

        def _inside(self, dist):
            if self.__inclusive:
                return dist <= self.__eps
            return dist < self.__eps

        def _search_iter(self):
            stack = [(0, len(self.__dataset))]
            result = []
            while stack:
                start, end = stack.pop()

                v_radius = self.__distances[start]
                v_point_index = self.__indices[start]
                v_point = self.__dataset[v_point_index]
                is_terminal = self.__is_terminal[start]

                if is_terminal:
                    for x_index in self.__indices[start:end]:
                        x = self.__dataset[x_index]
                        dist = self.__distance(self.__point, x)
                        if self._inside(dist):
                            result.append(x)
                else:
                    dist = self.__distance(self.__point, v_point)
                    mid = _mid(start, end)
                    if self._inside(dist):
                        result.append(v_point)
                    if dist <= v_radius:
                        fst = (start + 1, mid)
                        snd = (mid, end)
                    else:
                        fst = (mid, end)
                        snd = (start + 1, mid)
                    if abs(dist - v_radius) <= self.__eps:
                        stack.append(snd)
                    stack.append(fst)
            return result

    def knn_search(self, point, k):
        return self._KnnSearch(self, point, k).search()

    class _KnnSearch:

        def __init__(self, vpt, point, neighbors):
            self.__dataset = vpt._get_dataset()
            self.__distances = vpt._get_distances()
            self.__indices = vpt._get_indices()
            self.__is_terminal = vpt._get_is_terminal()
            self.__distance = vpt._get_distance()
            self.__point = point
            self.__neighbors = neighbors
            self.__radius = float("inf")
            self.__result = MaxHeap()

        def _get_items(self):
            while len(self.__result) > self.__neighbors:
                self.__result.pop()
            return [x for (_, x) in self.__result]

        def search(self):
            self._search_iter()
            return self._get_items()

        def _process(self, x):
            dist = self.__distance(self.__point, x)
            if dist >= self.__radius:
                return dist
            self.__result.add(dist, x)
            while len(self.__result) > self.__neighbors:
                self.__result.pop()
            if len(self.__result) == self.__neighbors:
                self.__radius, _ = self.__result.top()
            return dist

        def _search_iter(self):
            PRE, POST = 0, 1
            self.__result = MaxHeap()
            stack = [(0, len(self.__dataset), 0.0, PRE)]
            while stack:
                start, end, thr, action = stack.pop()

                v_radius = self.__distances[start]
                v_point_index = self.__indices[start]
                v_point = self.__dataset[v_point_index]
                is_terminal = self.__is_terminal[start]

                if is_terminal:
                    for x_index in self.__indices[start:end]:
                        x = self.__dataset[x_index]
                        self._process(x)
                else:
                    if action == PRE:
                        mid = _mid(start, end)
                        dist = self._process(v_point)
                        if dist <= v_radius:
                            fst_start, fst_end = start + 1, mid
                            snd_start, snd_end = mid, end
                        else:
                            fst_start, fst_end = mid, end
                            snd_start, snd_end = start + 1, mid
                        stack.append((snd_start, snd_end, abs(v_radius - dist), POST))
                        stack.append((fst_start, fst_end, 0.0, PRE))
                    elif action == POST:
                        if self.__radius > thr:
                            stack.append((start, end, 0.0, PRE))
            return self._get_items()
