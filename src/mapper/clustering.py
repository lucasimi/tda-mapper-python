"""A collection of clustering functions"""
from sklearn.cluster import DBSCAN, AffinityPropagation

def _from_sklearn(data, balls, algo):
    """Return sklearn clustering for every ball"""
    clusters = []
    for ball in balls:
        ball_arr = [data[x] for x in ball]
        local_clusters = algo.fit(ball_arr)
        cluster = list(zip(ball, local_clusters.labels_))
        clusters.append([(x, y) for (x, y) in cluster if y != -1])
    return clusters

def dbscan(data, balls, **kwargs):
    """Return dbscan clusters for every ball"""
    return _from_sklearn(data, balls, DBSCAN(**kwargs))

def affinity_propagation(data, balls, **kwargs):
    """Return affinity propagation clusters for every ball"""
    return _from_sklearn(data, balls, AffinityPropagation(**kwargs))

def trivial(data, balls, **kwargs):
    """Return a single cluster for every ball"""
    clusters = []
    for ball in balls:
        clusters.append([(x, 1) for x in ball])
    return clusters
