"""A collection of functions to build open covers"""
from .utils.balltree import BallTree

def _pullback_pseudometric(data, lens, metric):
    return lambda i, j : metric(lens(data[i]), lens(data[j]))

def _cover_fast(data, find_neighbors):
    non_covered = {i for i in range(len(data))}
    groups = []
    while non_covered:
        point = non_covered.pop()
        neighbors = find_neighbors(point)
        non_covered.difference_update(neighbors)
        if neighbors:
            groups.append(neighbors)
    return groups

class BallCover:

    def __init__(self, radius):
        self.__radius = radius

    def cover(self, data, metric, lens):
        metric_pb = _pullback_pseudometric(data, lens, metric)
        data_ids = [x for x in range(len(data))]
        dt = BallTree(metric_pb, data_ids, min_radius=self.__radius)
        find_neighbors = lambda p : dt.ball_search(p, self.__radius)
        return _cover_fast(data, find_neighbors)


class KnnCover:

    def __init__(self, k):
        self.__k = k

    def cover(self, data, metric, lens):
        metric_pb = _pullback_pseudometric(data, lens, metric)
        data_ids = [x for x in range(len(data))]
        dt = BallTree(metric_pb, data_ids, max_count=self.__k)
        find_neighbors = lambda p : dt.knn_search(p, self.__k)
        return _cover_fast(data, find_neighbors)


class TrivialCover:

    def cover(self, data, metric, lens):
        """Return a trivial grouping of data with a single ball"""
        return [[x for x in range(len(data))]]
