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

def ball_cover(data, metric, lens, radius):
    metric_pb = _pullback_pseudometric(data, lens, metric)
    data_ids = [x for x in range(len(data))]
    dt = BallTree(metric_pb, data_ids, min_radius=radius)
    find_neighbors = lambda p : dt.ball_search(p, radius)
    return _cover_fast(data, find_neighbors)

def knn_cover(data, metric, lens, k):
    metric_pb = _pullback_pseudometric(data, lens, metric)
    data_ids = [x for x in range(len(data))]
    dt = BallTree(metric_pb, data_ids, max_count=k)
    find_neighbors = lambda p : dt.knn_search(p, k)
    return _cover_fast(data, find_neighbors)

def trivial_cover(data, metric, lens):
    """Return a trivial grouping of data with a single ball"""
    return [[x for x in range(len(data))]]