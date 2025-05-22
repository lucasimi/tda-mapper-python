from tdamapper.utils.quickselect import quickselect, swap_all


def _mid(start, end):
    return (start + end) // 2


class VPArray:

    def __init__(self, dataset, distances, indices, is_terminal):
        self._dataset = dataset
        self._distances = distances
        self._indices = indices
        self._is_terminal = is_terminal

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

    def set_terminal(self, i, terminal):
        self._is_terminal[i] = terminal

    def is_terminal(self, i):
        return self._is_terminal[i]

    def swap(self, i, j):
        swap_all(self._distances, i, j, self._indices, self._is_terminal)

    def partition(self, s, e, k):
        quickselect(self._distances, s, e, k, self._indices, self._is_terminal)
