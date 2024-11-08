"""
This module provides functionalities to visualize the Mapper graph based on
matplotlib.
"""

import math

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tdamapper.core import (
    aggregate_graph,
    ATTR_SIZE,
)


_NODE_OUTER_WIDTH = 0.75

_NODE_OUTER_COLOR = '#777'

_EDGE_WIDTH = 0.75

_EDGE_COLOR = '#777'


def plot_matplotlib(
    mapper_plot,
    width,
    height,
    title,
    colors,
    agg,
    cmap,
):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(width * px, height * px))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    _plot_edges(mapper_plot, ax)
    _plot_nodes(mapper_plot, title, colors, agg, cmap, ax)
    return fig, ax


def _plot_nodes(mapper_plot, title, colors, agg, cmap, ax):
    nodes_arr = _node_pos_array(
        mapper_plot.graph,
        mapper_plot.dim,
        mapper_plot.positions
    )
    attr_size = nx.get_node_attributes(mapper_plot.graph, ATTR_SIZE)
    max_size = max(attr_size.values(), default=1.0)
    colors_agg = aggregate_graph(colors, mapper_plot.graph, agg)
    marker_color = [colors_agg[n] for n in mapper_plot.graph.nodes()]
    marker_size = [200.0 * math.sqrt(attr_size[n] / max_size) for n in mapper_plot.graph.nodes()]
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
        linewidths=_NODE_OUTER_WIDTH)
    colorbar = plt.colorbar(
        verts,
        orientation='vertical',
        location='right',
        aspect=40,
        pad=0.02,
        ax=ax,
        format="%.2g")
    colorbar.set_label(
        title,
        color=_NODE_OUTER_COLOR)
    colorbar.set_alpha(1.0)
    colorbar.outline.set_color(_NODE_OUTER_COLOR)
    colorbar.ax.yaxis.set_tick_params(
        color=_NODE_OUTER_COLOR,
        labelcolor=_NODE_OUTER_COLOR)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.ax.locator_params(nbins=10)


def _plot_edges(mapper_plot, ax):
    segments = [(mapper_plot.positions[e[0]], mapper_plot.positions[e[1]]) for e in mapper_plot.graph.edges()]
    lines = LineCollection(
        segments,
        color=_EDGE_COLOR,
        linewidth=_EDGE_WIDTH,
        alpha=1.0,
        zorder=-1,
        antialiased=True)
    ax.add_collection(lines)


def _node_pos_array(graph, dim, positions):
    return tuple([positions[n][i] for n in graph.nodes()] for i in range(dim))
