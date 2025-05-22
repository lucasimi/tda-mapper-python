class BallSearch:

    def __init__(self, vpt, point, eps, inclusive=True):
        self.__tree = vpt._get_tree()
        self._arr = vpt._get_arr()
        self.__distance = vpt._get_distance()
        self.__point = point
        self.__eps = eps
        self.__inclusive = inclusive
        self.__result = []

    def search(self):
        self.__result.clear()
        self._search_rec(self.__tree)
        return self.__result

    def _inside(self, dist):
        if self.__inclusive:
            return dist <= self.__eps
        return dist < self.__eps

    def _search_rec(self, tree):
        if tree.is_terminal():
            start, end = tree.get_bounds()
            for x in self._arr.get_points(start, end):
                dist = self.__distance(self.__point, x)
                if self._inside(dist):
                    self.__result.append(x)
        else:
            v_radius, v_point = tree.get_ball()
            dist = self.__distance(v_point, self.__point)
            if self._inside(dist):
                self.__result.append(v_point)
            if dist <= v_radius:
                fst, snd = tree.get_left(), tree.get_right()
            else:
                fst, snd = tree.get_right(), tree.get_left()
            self._search_rec(fst)
            if abs(dist - v_radius) <= self.__eps:
                self._search_rec(snd)
