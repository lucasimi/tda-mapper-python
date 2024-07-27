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
            self.__tree = self.__build_rec(set(ids), set(), 0, i)

    def __build_rec(self, ids, cov_ids, p_idx, i):
        p = self.__X[p_idx]
        near_ids = {j for j in ids if self.__metric(p, self.__X[j]) < 2**(i - 1)}
        far_ids = {j for j in ids if self.__metric(p, self.__X[j]) >= 2**(i - 1)}
        if len(ids) < 2:
            cov_ids.add(p_idx)
            return _Node(i, p)
        else:
            t = self.__build_rec(near_ids, cov_ids, p_idx, i - 1)
            c = [t]
            far_ids.difference_update(cov_ids)
            while far_ids:
                q_idx = far_ids.pop()
                q = self.__X[q_idx]
                near_ids = {j for j, x in enumerate(self.__X) if self.__metric(q, x) < 2**(i - 1)}
                near_ids.difference_update(cov_ids)
                t1 = self.__build_rec(near_ids, cov_ids, q_idx, i - 1)
                c.append(t1)
                far_ids.difference_update(cov_ids)
            return _Node(i, p, c)

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



        
        