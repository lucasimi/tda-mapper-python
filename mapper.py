import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import time

def no_warnings(*args, **kwargs):
    pass

import warnings
warnings.warn = no_warnings


class OpenCover:

    def __init__(self, distance, radius):
        self._distance = distance
        self._covered_points = set()
        self._radius = radius
        self._charts = {}

    def cover(self, dataset, lens):
        self._dataset = dataset
        self._lens = lambda x: lens[x] if isinstance(lens, list) else lens(dataset[x])
        for p, _ in enumerate(self._dataset):
            if p not in self._covered_points:
                self.__cover_point(p)
        return [LocalChart(dataset, self._charts[x]) for x in self._charts]

    def __is_point_in_chart(self, point, chart, radius):
        point_value = self._lens(point)
        chart_value = self._lens(chart)
        return self._distance(point_value, chart_value) < radius
    
    def __cover_point(self, point):
        for chart in self._charts:
            if self.__is_point_in_chart(point, chart, self._radius):
                self._charts[chart].append(point)
                self._covered_points.add(point)
        if point not in self._covered_points:
            self.__build_open_chart(point)
            self._covered_points.add(point)
            
    def __build_open_chart(self, point):
        local_chart = [point]
        for chart, points in self._charts.items():
            if self.__is_point_in_chart(point, chart, 2 * self._radius):
                for p in points:
                    if self.__is_point_in_chart(p, point, self._radius):
                        local_chart.append(p)
        self._charts[point] = local_chart
        

class LocalChart:
    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = indices
        
    def get_dataset(self):
        return self._dataset

    def get_points(self):
        return [self._dataset[x] for x in self._indices]
    
    def get_indices(self):
        return self._indices

    def perform_clustering(self, alg):
        points = self.get_points()
        labels = alg.fit(points).labels_
        clusters_dict = {}
        for idx, l in zip(self._indices, labels):
            if l not in clusters_dict:
                clusters_dict[l] = []
            clusters_dict[l].append(idx)
        clusters = [LocalCluster(self, ids) for ids in clusters_dict.values()]
        return clusters

    
class LocalCluster:
    def __init__(self, local_chart, indices):
        self._indices = indices
        self._local_chart = local_chart
        self._size = len(indices)

    def get_indices(self):
        return self._indices

    def get_points(self):
        return [self._local_chart.get_dataset()[x] for x in self._indices]
    
    def get_local_chart(self):
        return self._local_chart
        
    def get_size(self):
        return self._size

    def get_color(self, aggfunc, colors):
        if isinstance(colors, list):
            return aggfunc([colors[x] for x in self._indices])
        else:
            return aggfunc([colors(x) for x in self.get_points()])

        
class ClusteringGraph:
    def __init__(self, clustering):
        self._graph = {}
        self._clusters = clustering
        point_clusters = {}
        for n, loc_cluster in enumerate(clustering):
            self._graph[n] = (len(loc_cluster.get_indices()), {})
            for idx in loc_cluster.get_indices():
                if idx not in point_clusters:
                    point_clusters[idx] = set()
                point_clusters[idx].add(n)
            
        for idx, clusters in point_clusters.items():
            for c1 in clusters:
                for c2 in clusters:
                    if c1 == c2: continue
                    if c2 not in self._graph[c1][1]:
                        self._graph[c1][1][c2] = 0
                    else:
                        w = self._graph[c1][1][c2]
                        self._graph[c1][1][c2] = w + 1

    def __get_nodes(self):
        return list(self._graph.keys())
                        
    def __get_size(self, idx):
        return self._graph[idx][0]

    def __get_color(self, aggfunc, colors, idx):
        return self._clusters[idx].get_color(aggfunc, colors)
    
    def __get_adj(self, idx):
        return list(self._graph[idx][1].keys())

    def __get_weight(self, id1, id2):
        return self._graph[id1][1][id2]
    
    def plot(self, aggfunc, colors, colormap):
        g = nx.Graph()
        max_weight, max_size = 1.0, 1.0
        for u in self.__get_nodes():
            u_size = self.__get_size(u)
            u_color = colormap(self.__get_color(aggfunc, colors, u))
            g.add_node(u, size=u_size, color=u_color)
            if u_size > max_size:
                max_size = u_size
            for v in self.__get_adj(u):
                uv_weight = self.__get_weight(u, v)
                g.add_edge(u, v, weight=uv_weight)
                if uv_weight > max_weight:
                    max_weight = uv_weight
        sizes_dict = nx.get_node_attributes(g, 'size')
        sizes = [1.0 + 150.0 * sizes_dict[u] / max_size for u in g.nodes()]
        colors_dict = nx.get_node_attributes(g, 'color')
        colors_list = [colors_dict[u] for u in g.nodes()]
        weights = [3.0 * g[u][v]['weight'] / max_weight for u,v in g.edges()]
        nx.draw(g, width=weights, node_size=sizes, node_color=colors_list, edgecolors='black', linewidths=0.2, edge_color='black')
        plt.show()

        
class Mapper:

    def __init__(self, distance, radius):
        self._distance = distance
        self._radius = radius

    def fit(self, dataset, lens, alg):
        self.dataset = dataset
        
        t = time.time()
        self.open_cover = OpenCover(self._distance, self._radius)
        charts = self.open_cover.cover(dataset, lens)
        print(f'built open cover with {len(charts)} local chart(s)\n{time.time() - t}\n')

        t = time.time()
        clusters = []
        for chart in charts:
            for cluster in chart.perform_clustering(alg):
                clusters.append(cluster)
        self._clusters = clusters
        print(f'performed clustering with {len(clusters)} local cluster(s)\n{time.time() - t}\n')

        t = time.time()
        self._graph = ClusteringGraph(clusters)
        print(f'merged local data\n{time.time() - t}\n')

    def get_clusters(self):
        return self._clusters
        
    def plot(self, aggfunc, colors, colormap):
        self._graph.plot(aggfunc, colors, colormap)

        
class BallMapper(Mapper):
    def fit(self, dataset, alg):
        super().fit(dataset, lambda x: x, alg)        
    
