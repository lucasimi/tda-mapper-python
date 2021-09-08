"""A collection of clustering functions"""

def fit(data, balls, algo=None):
    """Perform clustering for every open cover"""
    clusters = []
    for ball in balls:
        if algo:
            ball_arr = [data[x] for x in ball]
            local_clusters = algo.fit(ball_arr)
            cluster = list(zip(ball, local_clusters.labels_))
            clusters.append([(x, y) for (x, y) in cluster if y != -1])
        else:
            clusters.append([(x, 1) for x in ball])
    return clusters
