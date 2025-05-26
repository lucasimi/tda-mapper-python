from tdamapper.utils.quickselect import quickselect, swap_all


def _mid(start, end):
    return (start + end) // 2


class VPArray:

    def __init__(self, dataset, distances, indices):
        self._dataset = dataset
        self._distances = distances
        self._indices = indices

    def size(self):
        return len(self._dataset)

    def get_point(self, i):
        return self._dataset[self._indices[i]]

    def get_points(self, s, e):
        for x_index in self._indices[s:e]:
            yield self._dataset[x_index]

    def get_distance(self, i):
        return self._distances[i]

    def set_distance(self, i, dist):
        self._distances[i] = dist

    def swap(self, i, j):
        swap_all(self._distances, i, j, self._indices)

    def partition(self, s, e, k):
        quickselect(self._distances, s, e, k, self._indices)


class Node:

    def __init__(self, radius, center, left, right):
        self._radius = radius
        self._center = center
        self._left = left
        self._right = right

    def get_ball(self):
        return self._radius, self._center

    def is_terminal(self):
        return False

    def get_left(self):
        return self._left

    def get_right(self):
        return self._right


class Leaf:

    def __init__(self, start, end):
        self._start = start
        self._end = end

    def get_bounds(self):
        return self._start, self._end

    def is_terminal(self):
        return True
