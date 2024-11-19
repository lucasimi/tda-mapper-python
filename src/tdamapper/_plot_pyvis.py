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
import plotly.colors as pc

from tdamapper.core import (
    aggregate_graph,
)


_EDGE_WIDTH = 0.75

_EDGE_COLOR = '#777'

_TICKS_NUM = 10


def __fmt(x, max_len=3):
    fmt = f'.{max_len}g'
    return f'{x:{fmt}}'


def _colorbar(height, cmap, cmin, cmax, title):
    colorbar_fig = go.Figure()
    colorbar_fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            showscale=True,
            reversescale=False,
            line_colorscale=cmap,
            colorscale=cmap,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                showticklabels=True,
                outlinewidth=1,
                borderwidth=0,
                orientation='v',
                thickness=20,
                thicknessmode='fraction',
                xanchor='left',
                titleside='right',
                tickwidth=1,
                tickformat='.2g',
                nticks=_TICKS_NUM,
                tickmode='auto',
                title=title,
            ),
        ),
    ))
    colorbar_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=100, t=0, b=0),
        width=80,
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return colorbar_fig


def _combine(network, colorbar):
    network_html = network.generate_html()
    colorbar_html = pio.to_html(
        colorbar,
        include_plotlyjs='cdn',
        full_html=False,
        config={
            'displayModeBar': False
        },
    )
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Network with Colorbar</title>
        <style>
            body {{
                margin: 0;
                height: 100vh;
                display: flex;
            }}

            .combined {{
                display: flex;
                flex-direction: row;
                margin: 0;
            }}

            .network {{
                display: flex;
                align-items: start;
            }}

            .colorbar {{
                display: flex;
                align-items: start;
            }}

            .network .card {{
                border-width: 0;
            }}

            .network #mynetwork {{
                border-width: 0;
            }}
        </style>
    </head>
    <body>
        <div class="combined">
            <div class="network">
                {network_html}
            </div>
            <div class="colorbar">
                {colorbar_html}
            </div>
        </div>
    </body>
    </html>
    """
    return combined_html


def plot_pyvis(
    mapper_plot,
    output_file,
    colors,
    agg,
    title,
    width,
    height,
    cmap,
):
    net, cmin, cmax = _compute_net(
        mapper_plot=mapper_plot,
        width=width,
        height=height,
        colors=colors,
        agg=agg,
        cmap=cmap,
    )
    colorbar = _colorbar(
        height=height,
        cmap=cmap,
        cmin=cmin,
        cmax=cmax,
        title=title
    )
    combined_html = _combine(net, colorbar)
    with open(output_file, 'w') as file:
        file.write(combined_html)


def _compute_net(
    mapper_plot,
    colors,
    agg,
    width,
    height,
    cmap,
):
    net = Network(
        height=f'{height}px',
        width=f'{width}px',
        directed=False,
        notebook=True,
        select_menu=False,
        filter_menu=False,
        neighborhood_highlight=True,
    )
    net.toggle_physics(False)
    graph = mapper_plot.graph
    nodes = graph.nodes
    cmap_colorscale = pc.get_colorscale(cmap)

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
        node_color = max(0.0, min(1.0, node_color))
        node_color_hex = pc.sample_colorscale(cmap_colorscale, node_color)[0]
        return node_color_hex

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
            y=-node_pos[1] * height,
        )

    for edge in graph.edges:
        source_id = int(edge[0])
        target_id = int(edge[1])
        edge_color = _EDGE_COLOR
        edge_width = _EDGE_WIDTH
        edge_width = 1.5
        net.add_edge(source_id, target_id, color=edge_color, width=edge_width)

    return net, min_node_color, max_node_color
