from tdamapper.utils.heap import MaxHeap


class KnnSearch:

    def __init__(self, vpt, point, neighbors):
        self.__tree = vpt._get_tree()
        self._arr = vpt._get_arr()
        self.__distance = vpt._get_distance()
        self.__point = point
        self.__neighbors = neighbors
        self.__items = MaxHeap()

    def _add(self, dist, x):
        self.__items.add(dist, x)
        if len(self.__items) > self.__neighbors:
            self.__items.pop()

    def _get_items(self):
        while len(self.__items) > self.__neighbors:
            self.__items.pop()
        return [x for (_, x) in self.__items]

    def _get_radius(self):
        if len(self.__items) < self.__neighbors:
            return float("inf")
        furthest_dist, _ = self.__items.top()
        return furthest_dist

    def search(self):
        self._search_rec(self.__tree)
        return self._get_items()

    def _search_rec(self, tree):
        if tree.is_terminal():
            start, end = tree.get_bounds()
            for x in self._arr.get_points(start, end):
                dist = self.__distance(self.__point, x)
                if dist < self._get_radius():
                    self._add(dist, x)
        else:
            v_radius, v_point = tree.get_ball()
            dist = self.__distance(v_point, self.__point)
            if dist < self._get_radius():
                self._add(dist, v_point)
            if dist <= v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            self._search_rec(fst)
            if abs(dist - v_radius) <= self._get_radius():
                self._search_rec(snd)
