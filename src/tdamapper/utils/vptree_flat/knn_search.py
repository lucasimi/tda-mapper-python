from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.vptree_flat.common import _mid

_PRE = 0
_POST = 1


class KnnSearch:

    def __init__(self, vpt, point, neighbors):
        self._arr = vpt._get_arr()
        self._distance = vpt._get_distance()
        self._point = point
        self._neighbors = neighbors
        self._radius = float("inf")
        self._result = MaxHeap()

    def _get_items(self):
        while len(self._result) > self._neighbors:
            self._result.pop()
        return [x for (_, x) in self._result]

    def search(self):
        self._search_iter()
        return self._get_items()

    def _process(self, x):
        dist = self._distance(self._point, x)
        if dist >= self._radius:
            return dist
        self._result.add(dist, x)
        while len(self._result) > self._neighbors:
            self._result.pop()
        if len(self._result) == self._neighbors:
            self._radius, _ = self._result.top()
        return dist

    def _search_iter(self):
        self._result = MaxHeap()
        stack = [(0, self._arr.size(), 0.0, _PRE)]
        while stack:
            start, end, thr, action = stack.pop()

            v_radius = self._arr.get_distance(start)
            v_point = self._arr.get_point(start)
            is_terminal = self._arr.is_terminal(start)

            if is_terminal:
                for x in self._arr.get_points(start, end):
                    self._process(x)
            else:
                if action == _PRE:
                    mid = _mid(start, end)
                    dist = self._process(v_point)
                    if dist <= v_radius:
                        fst_start, fst_end = start + 1, mid
                        snd_start, snd_end = mid, end
                    else:
                        fst_start, fst_end = mid, end
                        snd_start, snd_end = start + 1, mid
                    stack.append((snd_start, snd_end, abs(v_radius - dist), _POST))
                    stack.append((fst_start, fst_end, 0.0, _PRE))
                elif action == _POST:
                    if self._radius > thr:
                        stack.append((start, end, 0.0, _PRE))
        return self._get_items()
