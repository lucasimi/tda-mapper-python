import math


class _Node:

    def __init__(self, level, point, children=None):
        self.__level = level
        self.__point = point
        if children is None:
            self.__children = []
        else:
            self.__children = [x for x in children]

    def get_point(self):
        return self.__point

    def get_level(self):
        return self.__level

    def get_children(self):
        return self.__children


class CoverTree:

    def __init__(self, X, metric):
        self.__metric = metric
        self.__X = X
        self.__tree = None
        self.__build()

    def __build(self):
        if not self.__X:
            self.__tree = None
        else:
            p = self.__X[0]
            d = max([self.__metric(p, x) for x in self.__X])
            i = math.ceil(math.log2(d))
            ids = list(range(len(self.__X)))
            near = set(ids)
            near.remove(0)
            self.__tree, _ = self.__build_rec(0, i, near, set())

    def __split(self, p_idx, i, s):
        near = set()
        far = set()
        p = self.__X[p_idx]
        for j in s:
            x = self.__X[j]
            d = self.__metric(p, x)
            if d <= 2**i:
                near.add(j)
            elif d < 2**(i + 1):
                far.add(j)
        s.difference_update(near)
        s.difference_update(far)
        return near, far

    def __split2(self, p_idx, i, s1, s2):
        near = set()
        far = set()
        p = self.__X[p_idx]
        def _proc(j):
            x = self.__X[j]
            d = self.__metric(p, x)
            if d <= 2**i:
                near.add(j)
            elif d < 2**(i + 1):
                far.add(j)
        for j in s1:
            _proc(j)
        s1.difference_update(near)
        s1.difference_update(far)
        for j in s2:
            _proc(j)
        s2.difference_update(near)
        s2.difference_update(far)
        return near, far

    def __build_rec(self, p_idx, i, near_ids, far_ids):
        p = self.__X[p_idx]
        if not near_ids:
            return _Node(i, p), set()
        else:
            _n, _f = self.__split(p_idx, i - 1, near_ids)
            t, near_ids = self.__build_rec(p_idx, i - 1, _n, _f)
            children = [t]
            while near_ids:
                q_idx = near_ids.pop()
                _n, _f = self.__split2(q_idx, i - 1, near_ids, far_ids)
                t1, unused_ids = self.__build_rec(q_idx, i - 1, _n, _f)
                children.append(t1)
                new_near_ids, new_far_ids = self.__split(p_idx, i, unused_ids)
                near_ids.update(new_near_ids)
                far_ids.update(new_far_ids)
            return _Node(i, p, children), far_ids

    def get_tree(self):
        return self.__tree

    def get_max_level(self):
        return self.__tree.get_level()

    def get_level_nodes(self, i):
        return self._get_level_nodes(self.__tree, i)

    def _get_level_nodes(self, t, i):
        l = t.get_level()
        if l == i:
            return [t]
        elif l > i:
            p = []
            for c in t.get_children():
                p.extend(self._get_level_nodes(c, i))
            return p
        else:
            return []

    def get_level_points(self, i):
        return self._get_level_points(self.__tree, i)

    def _get_level_points(self, t, i):
        l = t.get_level()
        if l == i:
            return [t.get_point()]
        elif l > i:
            p = []
            for c in t.get_children():
                p.extend(self._get_level_points(c, i))
            return p
        else:
            return []

    def get_min_level(self):
        return self._get_min_level(self.__tree)

    def _get_min_level(self, t):
        l = t.get_level()
        c = t.get_children()
        if not c:
            return l
        else:
            return min([self._get_min_level(u) for u in c])
