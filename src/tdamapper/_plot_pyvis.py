"""
This module provides functionalities to visualize the Mapper graph based on
pyvis.
"""
import math

from pyvis.network import Network

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import plotly.graph_objects as go
import plotly.io as pio

from tdamapper.core import (
    aggregate_graph,
)


_EDGE_WIDTH = 0.75

_EDGE_COLOR = '#777'


def __fmt(x, max_len=3):
    fmt = f'.{max_len}g'
    return f'{x:{fmt}}'


def _colorbar(width, height, cmap, cmin, cmax, title):
    colorbar_fig = go.Figure()
    colorbar_fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale=cmap,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=title,
                thickness=20,
                len=1.0,
            ),
        ),
    ))
    colorbar_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=50, r=50, t=0, b=0),
        #width=width,
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return colorbar_fig


def _combine(network, colorbar):
    network_html = network.generate_html()
    colorbar_html = pio.to_html(colorbar, include_plotlyjs='cdn', full_html=False)
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Network with Colorbar</title>
        <style>
            body {{
                display: flex;
                flex-direction: row;
                margin: 0;
                padding: 0;
                height: 100vh;
            }}
            .network {{
                flex: 3;
            }}
        </style>
    </head>
    <body>
        <div class="network">
            {network_html}
        </div>
        <div class="colorbar">
            {colorbar_html}
        </div>
    </body>
    </html>
    """
    return combined_html


def plot_pyvis(
    mapper_plot,
    notebook,
    output_file,
    colors,
    agg,
    title,
    width,
    height,
    cmap,
):
    net = _compute_net(
        mapper_plot=mapper_plot,
        width=width,
        height=height,
        colors=colors,
        agg=agg,
        cmap=cmap,
        notebook=notebook,
    )
    net.show(output_file, notebook=notebook)


def _compute_net(
    mapper_plot,
    notebook,
    colors,
    agg,
    width,
    height,
    cmap,
):
    net = Network(
        height=height,
        width=width,
        directed=False,
        notebook=notebook,
        select_menu=True,
        filter_menu=True,
        neighborhood_highlight=True,
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
        if max_node_size == min_node_size:
            node_size_norm = 25.0
        else:
            node_size = int(nodes[node]['size'])
            node_size_norm = 25.0 * math.sqrt(node_size / max_node_size)
        return int(round(node_size_norm))

    def _color(node):
        if max_node_color == min_node_color:
            node_color = 0.5
        else:
            node_color = node_colors[node]
            node_color = (node_color - min_node_color) / (max_node_color - min_node_color)
        node_color = colormap(node_color)
        return mcolors.to_hex(node_color)

    def _blend_color(source, target):
        if max_node_color == min_node_color:
            blend_color = 0.5
        else:
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
        node_stats = __fmt(node_colors[node])
        node_label = f'color: {node_stats}\nnode: {node_id}\nsize: {node_size}'
        node_pos = mapper_plot.positions[node]
        net.add_node(
            node_id,
            label=node_id,
            size=node_size,
            color=node_color,
            title=node_label,
            x=node_pos[0] * width,
            y=node_pos[1] * height,
        )

    for edge in graph.edges:
        source_id = int(edge[0])
        target_id = int(edge[1])
        #edge_color = _blend_color(edge[0], edge[1])
        edge_color = _EDGE_COLOR
        edge_width = _EDGE_WIDTH
        edge_width = 1.5
        net.add_edge(source_id, target_id, color=edge_color, width=edge_width)

    return net
