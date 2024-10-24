from pyvis.network import Network

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tdamapper.core import (
    aggregate_graph,
)


def plot_pyvis(
            mapper_plot,
            width,
            height,
            colors,
            agg,
            cmap,
            notebook,
            output_file
        ):
    net = _compute_net(
        mapper_plot,
        width,
        height,
        colors,
        agg,
        cmap,
        notebook,
    )
    net.show(output_file)


def _compute_net(
            mapper_plot,
            width,
            height,
            colors,
            agg,
            cmap,
            notebook,
        ):
    net = Network(
        height=height,
        width=width,
        directed=False,
        notebook=notebook,
        select_menu=True,
        filter_menu=True,
        neighborhood_highlight=True
    )
    net.toggle_physics(False)
    graph = mapper_plot.graph
    nodes = graph.nodes

    min_node_size = float('inf')
    max_node_size = -float('inf')
    for node in nodes:
        node_size = nodes[node]['size']
        if node_size > max_node_size:
            max_node_size = node_size
        if node_size < min_node_size:
            min_node_size = node_size

    node_colors = aggregate_graph(colors, graph, agg)
    colormap = plt.get_cmap(cmap)

    min_node_color = float('inf')
    max_node_color = -float('inf')
    for node in nodes:
        node_color = node_colors[node]
        if node_color > max_node_color:
            max_node_color = node_color
        if node_color < min_node_color:
            min_node_color = node_color

    def _size(node):
        node_size = int(nodes[node]['size'])
        node_size_norm = (node_size - min_node_size) / (max_node_size - min_node_size) * 25
        return int(round(node_size_norm))

    def _color(node):
        node_color = node_colors[node]
        node_color = (node_color - min_node_color) / (max_node_color - min_node_color)
        node_color = colormap(node_color)
        return mcolors.to_hex(node_color)

    def _blend_color(source, target):
        source_color = node_colors[source]
        source_color = (source_color - min_node_color) / (max_node_color - min_node_color)
        target_color = node_colors[target]
        target_color = (target_color - min_node_color) / (max_node_color - min_node_color)
        blend_color = (source_color + target_color) / 2.0
        blend_color = colormap(blend_color)
        return mcolors.to_hex(blend_color)

    for node in nodes:
        node_id = int(node)
        node_size = _size(node)
        node_color = _color(node)
        node_pos = mapper_plot.positions[node]
        net.add_node(
            node_id,
            label=node_id,
            size=node_size,
            color=node_color,
            x=node_pos[0] * 1000.0,
            y=node_pos[1] * 1000.0,
        )

    for edge in graph.edges:
        source_id = int(edge[0])
        target_id = int(edge[1])
        edge_color = _blend_color(edge[0], edge[1])
        net.add_edge(source_id, target_id, color=edge_color)

    return net
