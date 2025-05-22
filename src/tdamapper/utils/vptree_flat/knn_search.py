from tdamapper.utils.heap import MaxHeap
from tdamapper.utils.vptree_flat.common import _mid


class KnnSearch:

    def __init__(self, vpt, point, neighbors):
        self._arr = vpt._get_arr()
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
        stack = [(0, self._arr.size(), 0.0, PRE)]
        while stack:
            start, end, thr, action = stack.pop()

            v_radius = self._arr.get_distance(start)
            v_point = self._arr.get_point(start)
            is_terminal = self._arr.is_terminal(start)

            if is_terminal:
                for x in self._arr.get_points(start, end):
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
