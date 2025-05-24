from tdamapper.utils.vptree_flat.common import _mid


class BallSearch:

    def __init__(self, vpt, point, eps, inclusive=True):
        self._arr = vpt._get_arr()
        self._distance = vpt._get_distance()
        self._point = point
        self._eps = eps
        self._inclusive = inclusive

    def search(self):
        return self._search_iter()

    def _inside(self, dist):
        if self._inclusive:
            return dist <= self._eps
        return dist < self._eps

    def _search_iter(self):
        stack = [(0, self._arr.size())]
        result = []
        while stack:
            start, end = stack.pop()
            v_radius = self._arr.get_distance(start)
            v_point = self._arr.get_point(start)
            is_terminal = self._arr.is_terminal(start)
            if is_terminal:
                for x in self._arr.get_points(start, end):
                    dist = self._distance(self._point, x)
                    if self._inside(dist):
                        result.append(x)
            else:
                dist = self._distance(self._point, v_point)
                mid = _mid(start, end)
                if self._inside(dist):
                    result.append(v_point)
                if dist <= v_radius:
                    fst = (start + 1, mid)
                    snd = (mid, end)
                else:
                    fst = (mid, end)
                    snd = (start + 1, mid)
                if abs(dist - v_radius) <= self._eps:
                    stack.append(snd)
                stack.append(fst)
        return result
