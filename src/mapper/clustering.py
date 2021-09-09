"""A collection of clustering functions"""


class ClusteringAlgorithm:

    def __init__(self, algo):
        self.__algo = algo

    def __get_clusters(self, data, ids):
        local_data = [data[x] for x in ids]
        local_clusters = self.__algo.fit(local_data)
        return list(zip(ids, local_clusters.labels_))

    def fit(self, data, atlas_ids):
        clusters = []
        for chart_ids in atlas_ids:
            chart_data = [data[x] for x in chart_ids]
            chart_clusters = self.__get_clusters(data, chart_data)
            clusters.append([(x, y) for (x, y) in chart_clusters if y != -1])
        return clusters


class TrivialClustering:

    def fit(self, data, atlas_ids):
        return [[(x, 1) for x in chart_ids] for chart_ids in atlas_ids]

