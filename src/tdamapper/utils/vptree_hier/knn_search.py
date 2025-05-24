from tdamapper.utils.heap import MaxHeap


class KnnSearch:

    def __init__(self, vpt, point, neighbors):
        self._tree = vpt._get_tree()
        self._arr = vpt._get_arr()
        self._distance = vpt._get_distance()
        self._point = point
        self._neighbors = neighbors
        self._items = MaxHeap()

    def _add(self, dist, x):
        self._items.add(dist, x)
        if len(self._items) > self._neighbors:
            self._items.pop()

    def _get_items(self):
        while len(self._items) > self._neighbors:
            self._items.pop()
        return [x for (_, x) in self._items]

    def _get_radius(self):
        if len(self._items) < self._neighbors:
            return float("inf")
        furthest_dist, _ = self._items.top()
        return furthest_dist

    def search(self):
        self._search_rec(self._tree)
        return self._get_items()

    def _search_rec(self, tree):
        if tree.is_terminal():
            start, end = tree.get_bounds()
            for x in self._arr.get_points(start, end):
                dist = self._distance(self._point, x)
                if dist < self._get_radius():
                    self._add(dist, x)
        else:
            v_radius, v_point = tree.get_ball()
            dist = self._distance(v_point, self._point)
            if dist < self._get_radius():
                self._add(dist, v_point)
            if dist <= v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            self._search_rec(fst)
            if abs(dist - v_radius) <= self._get_radius():
                self._search_rec(snd)
