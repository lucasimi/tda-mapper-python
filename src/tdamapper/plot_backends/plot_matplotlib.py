"""
This module provides functionalities to visualize the Mapper graph based on
matplotlib.
"""

import math
from typing import Any, Callable, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from numpy.typing import NDArray

from tdamapper.core import ATTR_SIZE, aggregate_graph
from tdamapper.plot_backends.common import MapperPlotType

_NODE_OUTER_WIDTH = 0.75

_NODE_OUTER_COLOR = "#777"

_EDGE_WIDTH = 0.75

_EDGE_COLOR = "#777"


def plot_matplotlib(
    mapper_plot: MapperPlotType,
    width: int,
    height: int,
    title: Optional[str],
    colors: NDArray[np.float_],
    node_size: Union[float, int],
    agg: Callable[..., Any],
    cmap: str,
) -> tuple[Figure, Axes]:
    """
    Draw a static plot using Matplotlib.

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

    :return: A static matplotlib figure that can be displayed on screen
        and notebooks.
    """
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, ax = plt.subplots(figsize=(width * px, height * px))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    _plot_edges(mapper_plot, ax)
    _plot_nodes(mapper_plot, title, colors, node_size, agg, cmap, ax)
    return fig, ax


def _plot_nodes(
    mapper_plot: MapperPlotType,
    title: Optional[str],
    colors: NDArray[np.float_],
    node_size: Union[int, float],
    agg: Callable[..., Any],
    cmap: str,
    ax: Axes,
) -> None:
    nodes_arr = _node_pos_array(
        mapper_plot.graph, mapper_plot.dim, mapper_plot.positions
    )
    attr_size = nx.get_node_attributes(mapper_plot.graph, ATTR_SIZE)
    max_size = max(attr_size.values(), default=1.0)
    colors_agg = aggregate_graph(colors, mapper_plot.graph, agg)
    marker_color = [colors_agg[n] for n in mapper_plot.graph.nodes()]
    marker_size = [
        node_size * 200.0 * math.sqrt(attr_size[n] / max_size)
        for n in mapper_plot.graph.nodes()
    ]
    verts = ax.scatter(
        x=nodes_arr[0],
        y=nodes_arr[1],
        c=marker_color,
        s=marker_size,
        cmap=cmap,
        alpha=1.0,
        vmin=min(marker_color, default=None),
        vmax=max(marker_color, default=None),
        edgecolors=_NODE_OUTER_COLOR,
        linewidths=_NODE_OUTER_WIDTH,
    )
    colorbar = plt.colorbar(
        verts,
        orientation="vertical",
        location="right",
        aspect=40,
        pad=0.02,
        ax=ax,
        format="%.2g",
    )
    if title is not None:
        colorbar.set_label(title, color=_NODE_OUTER_COLOR)
    colorbar.set_alpha(1.0)
    # colorbar.outline.set_color(_NODE_OUTER_COLOR)
    colorbar.ax.yaxis.set_tick_params(
        color=_NODE_OUTER_COLOR, labelcolor=_NODE_OUTER_COLOR
    )
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.locator_params(nbins=10)


def _plot_edges(mapper_plot: MapperPlotType, ax: Axes) -> None:
    segments = [
        (mapper_plot.positions[e[0]], mapper_plot.positions[e[1]])
        for e in mapper_plot.graph.edges()
    ]
    lines = LineCollection(
        segments,
        color=_EDGE_COLOR,
        linewidth=_EDGE_WIDTH,
        alpha=1.0,
        zorder=-1,
        antialiased=True,
    )
    ax.add_collection(lines)


def _node_pos_array(
    graph: nx.Graph, dim: int, positions: dict[int, tuple[float, ...]]
) -> tuple[list[float], ...]:
    return tuple([positions[n][i] for n in graph.nodes()] for i in range(dim))
