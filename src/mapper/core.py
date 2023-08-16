import networkx as nx


_ATTR_IDS = 'ids'
_ATTR_SIZE = 'size'

_ID_IDS = 0
_ID_NEIGHS = 1


def build_labels(X, y, cover, clustering):
    '''
    Takes a dataset, returns a list of lists, where the list at position i
    contains the cluster ids to which the item at position i belongs to.
    * Each list in the output is a sorted list of ints with no duplicate.
    '''
    max_label = 0
    labels = [[] for _ in X]
    for neigh_ids in cover.charts(y):
        neigh_data = [X[j] for j in neigh_ids]
        try:
            neigh_labels = clustering.fit(neigh_data).labels_
        except:
            neigh_labels = [0 for _ in neigh_data]
        max_neigh_label = 0
        for (neigh_id, neigh_label) in zip(neigh_ids, neigh_labels):
            if neigh_label != -1:
                if neigh_label > max_neigh_label:
                    max_neigh_label = neigh_label
                labels[neigh_id].append(max_label + neigh_label)
        max_label += max_neigh_label + 1
    return labels
        

def build_adjaciency(labels):
    '''
    Takes a list of lists of items, returns a dict where each item is
    mapped to a couple. Inside each couple the first entry is the list 
    of positions where the item is present, the second entry is the list
    of items which appear in any of the lists where the key is present.
    '''
    adj = {}
    for n, clusters in enumerate(labels):
        for label in clusters:
            if label not in adj:
                adj[label] = ([], [])
            adj[label][_ID_IDS].append(n)
    edges = set()
    for clusters in labels:
        clusters_len = len(clusters)
        for i in range(clusters_len):
            source = clusters[i]
            for j in range(i + 1, clusters_len):
                target = clusters[j]
                if (source, target) not in edges:
                    target = clusters[j]
                    adj[source][_ID_NEIGHS].append(target)
                    edges.add((source, target))
                    adj[target][_ID_NEIGHS].append(source)
                    edges.add((target, source))
    return adj


def build_graph(X, y, cover, clustering):
    labels = build_labels(X, y, cover, clustering)
    adjaciency = build_adjaciency(labels)
    graph = nx.Graph()
    for source, (items, _) in adjaciency.items():
        graph.add_node(source, **{_ATTR_SIZE: len(items), _ATTR_IDS: items})
    edges = set()
    for source, (_, target_ids) in adjaciency.items():
        for target in target_ids:
            if (source, target) not in edges:
                graph.add_edge(source, target)
                edges.add((source, target))
                graph.add_edge(target, source)
                edges.add((target, source))
    return graph


def build_charts(X, search):
    covered = set()
    search.fit(X)
    for i, xi in enumerate(X):
        if i not in covered:
            neigh_ids = search.neighbors(xi)
            covered.update(neigh_ids)
            yield neigh_ids


def build_connected_components(graph):
    '''
    Takes a dataset and a graph, where each node represents a sets of elements
    from the dataset, returns a list of integers, where position i is the id
    of the connected component of the graph where the element at position i 
    from the dataset lies.
    '''
    cc_id = 1
    item_cc = {}
    for connected_component in nx.connected_components(graph):
        for node in connected_component:
            for item_id in graph.nodes[node][_ATTR_IDS]:
                item_cc[item_id] = cc_id
        cc_id += 1
    return item_cc


def compute_local_interpolation(y, graph, agg):
    agg_values = {}
    nodes = graph.nodes()
    for node_id in nodes:
        node_values = [y[i] for i in nodes[node_id][_ATTR_IDS]]
        agg_value = agg(node_values)
        agg_values[node_id] = agg_value
    return agg_values


class MapperAlgorithm:

    def __init__(self, cover, clustering):
        self.__cover = cover
        self.__clustering = clustering
        self.graph_ = None

    def fit(self, X, y=None):
        self.graph_ = self.fit_transform(X, y)
        return self
            
    def fit_transform(self, X, y):
        return build_graph(X, y, self.__cover, self.__clustering)


class MapperClassifier:

    def __init__(self, mapper_algo):
        self.__mapper_algo = mapper_algo
        self.labels_ = None

    def fit(self, X, y=None):
        self.labels_ = self.fit_predict(X, y)
        return self
            
    def fit_predict(self, X, y):
        graph = self.__mapper_algo.fit_transform(X, y)
        ccs = build_connected_components(graph)
        return [ccs[i] for i, _ in enumerate(X)]
