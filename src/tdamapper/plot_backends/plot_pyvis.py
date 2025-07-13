"""
This module provides functionalities to visualize the Mapper graph based on
pyvis.
"""

import math
from typing import Any, Callable, Optional, Union

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.io as pio
from numpy.typing import NDArray
from pyvis.network import Network

from tdamapper.core import aggregate_graph
from tdamapper.plot_backends.common import MapperPlotType

_EDGE_WIDTH = 0.75

_EDGE_COLOR = "#777"

_TICKS_NUM = 10


def _fmt(x: Any, max_len: int = 3) -> str:
    fmt = f".{max_len}g"
    return f"{x:{fmt}}"


def _colorbar(
    height: int,
    cmap: str,
    cmin: float,
    cmax: float,
    title: Optional[str],
) -> go.Figure:
    colorbar_fig = go.Figure()
    colorbar_fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
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
                    orientation="v",
                    thickness=20,
                    thicknessmode="fraction",
                    xanchor="left",
                    title_side="right",
                    tickwidth=1,
                    tickformat=".2g",
                    nticks=_TICKS_NUM,
                    tickmode="auto",
                    title=title,
                ),
            ),
        )
    )
    colorbar_fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=100, t=0, b=0),
        width=80,
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return colorbar_fig


def _combine(network: Network, colorbar: go.Figure) -> str:
    network_html = network.generate_html()
    colorbar_html = pio.to_html(
        colorbar,
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False},
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
    mapper_plot: MapperPlotType,
    output_file: str,
    colors: NDArray[np.float_],
    node_size: Union[float, int],
    agg: Callable[..., Any],
    title: Optional[str],
    width: int,
    height: int,
    cmap: str,
) -> None:
    """
    Draw an interactive HTML plot using PyVis.

    :param output_file: The path where the html file is written.
    :param colors: An array of values that determine the color of each
        node in the graph, useful for highlighting different features of
        the data.
    :param node_size: A scaling factor for node size.
    :param agg: A function used to aggregate the `colors` array over the
        points within a single node. The final color of each node is
        obtained by mapping the aggregated value with the colormap `cmap`.
    :param title: The title to be displayed alongside the figure.
    :param cmap: The name of a colormap used to map `colors` data values,
        aggregated by `agg`, to actual RGBA colors.
    :param width: The desired width of the figure in pixels.
    :param height: The desired height of the figure in pixels.
    """
    net, cmin, cmax = _compute_net(
        mapper_plot=mapper_plot,
        width=width,
        height=height,
        colors=colors,
        node_size=node_size,
        agg=agg,
        cmap=cmap,
    )
    colorbar = _colorbar(height=height, cmap=cmap, cmin=cmin, cmax=cmax, title=title)
    combined_html = _combine(net, colorbar)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(combined_html)


def _compute_net(
    mapper_plot: MapperPlotType,
    colors: NDArray[np.float_],
    node_size: Union[float, int],
    agg: Callable[..., Any],
    width: int,
    height: int,
    cmap: str,
) -> tuple[Network, float, float]:
    net = Network(
        height=f"{height}px",
        width=f"{width}px",
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

    min_node_size = float("inf")
    max_node_size = -float("inf")
    for node in nodes:
        n_size = nodes[node]["size"]
        if n_size > max_node_size:
            max_node_size = n_size
        if n_size < min_node_size:
            min_node_size = n_size

    node_colors = aggregate_graph(colors, graph, agg)

    min_node_color = float("inf")
    max_node_color = -float("inf")
    for node in nodes:
        node_color = node_colors[node]
        if node_color > max_node_color:
            max_node_color = node_color
        if node_color < min_node_color:
            min_node_color = node_color
    node_color_range = max_node_color - min_node_color

    def _size(node: int) -> int:
        if max_node_size == min_node_size:
            node_size_norm = 25.0
        else:
            n_size = int(nodes[node]["size"])
            node_size_norm = node_size * 25.0 * math.sqrt(n_size / max_node_size)
        return int(round(node_size_norm))

    def _color(node: int) -> Any:
        if max_node_color == min_node_color:
            node_color = 0.5
        else:
            node_color = node_colors[node]
            node_color = (node_color - min_node_color) / node_color_range
        node_color = max(0.0, min(1.0, node_color))
        node_color_hex = pc.sample_colorscale(cmap_colorscale, node_color)[0]
        return node_color_hex

    for node in nodes:
        node_id = int(node)
        n_size = _size(node)
        node_color = _color(node)
        node_stats = _fmt(node_colors[node])
        node_label = f"color: {node_stats}\nnode: {node_id}\nsize: {n_size}"
        node_pos = mapper_plot.positions[node]
        net.add_node(
            node_id,
            label=node_id,
            size=n_size,
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
