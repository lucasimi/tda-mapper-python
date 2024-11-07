"""
This module provides functionalities to visualize the Mapper graph based on
plotly.
"""

import math

import numpy as np

import networkx as nx

import plotly.graph_objects as go

from tdamapper.core import (
    ATTR_SIZE,
    aggregate_graph,
)


_NODE_OUTER_WIDTH = 0.75

_NODE_OUTER_COLOR = '#777'

_NODE_OPACITY = 1.0

_EDGE_WIDTH = 0.75

_EDGE_OPACITY = 1.0

_EDGE_COLOR = '#777'

_TICKS_NUM = 10


def _node_pos_array(graph, dim, node_pos):
    return tuple([node_pos[n][i] for n in graph.nodes()] for i in range(dim))


def _edge_pos_array(graph, dim, node_pos):
    edges_arr = tuple([] for i in range(dim))
    for edge in graph.edges():
        pos0, pos1 = node_pos[edge[0]], node_pos[edge[1]]
        for i in range(dim):
            edges_arr[i].append(pos0[i])
            edges_arr[i].append(pos1[i])
            edges_arr[i].append(None)
    return edges_arr


def plot_plotly(
    mapper_plot,
    width,
    height,
    title,
    colors,
    agg=np.nanmean,
    cmap='jet',
):
    node_col = aggregate_graph(colors, mapper_plot.graph, agg)
    fig = _figure(mapper_plot, node_col, width, height, title, cmap)
    return fig


def plot_plotly_update(
    mapper_plot,
    fig,
    width=None,
    height=None,
    title=None,
    colors=None,
    agg=None,
    cmap=None,
):
    if (colors is not None) and (agg is not None) and (cmap is not None):
        _update_traces_col(mapper_plot, fig, colors, agg, cmap)
    if cmap is not None:
        _update_traces_cmap(mapper_plot, fig, cmap)
    if title is not None:
        _update_traces_title(mapper_plot, fig, title)
    if (width is not None) and (height is not None):
        _update_layout(fig, width, height)
    return fig


def _update_traces_col(mapper_plot, fig, colors, agg, cmap):
    if (colors is not None) and (agg is not None):
        nodes_col = aggregate_graph(colors, mapper_plot.graph, agg)
        colors_list = list(nodes_col.values())
        _update_node_trace_col(mapper_plot, fig, nodes_col, colors_list)
        _update_edge_trace_col(mapper_plot, fig, cmap, nodes_col, colors_list)


def _update_edge_trace_col(mapper_plot, fig, cmap, colors_agg, colors_list):
    colors_avg = []
    for edge in mapper_plot.graph.edges():
        c0, c1 = colors_agg[edge[0]], colors_agg[edge[1]]
        colors_avg.append(c0)
        colors_avg.append(c1)
        colors_avg.append(c1)
    if not colors_avg:
        return
    if mapper_plot.dim == 3:
        fig.update_traces(
            patch=dict(
                line_color=colors_avg,
                line_colorscale=cmap,
                line_cmax=max(colors_list, default=None),
                line_cmin=min(colors_list, default=None),
            ),
            selector=dict(name='edges_trace'),
        )


def _update_node_trace_col(mapper_plot, fig, colors_agg, colors_list):
    fig.update_traces(
        patch=dict(
            text=_text(mapper_plot, colors_agg),
            marker_color=colors_list,
            marker_cmax=max(colors_list, default=None),
            marker_cmin=min(colors_list, default=None),
        ),
        selector=dict(name='nodes_trace'),
    )


def _update_traces_cmap(mapper_plot, fig, cmap):
    fig.update_traces(
        patch=dict(
            marker_colorscale=cmap,
            marker_line_colorscale=cmap,
        ),
        selector=dict(name='nodes_trace'),
    )
    if mapper_plot.dim == 3:
        fig.update_traces(
            patch=dict(line_colorscale=cmap),
            selector=dict(name='edges_trace'),
        )


def _update_traces_title(mapper_plot, fig, title):
    fig.update_traces(
        patch=dict(marker_colorbar=_colorbar(mapper_plot, title)),
        selector=dict(name='nodes_trace'),
    )


def _update_layout(fig, width, height):
    fig.update_layout(
        width=width,
        height=height,
    )


def _figure(mapper_plot, node_col, width, height, title, cmap):
    node_pos = mapper_plot.positions
    node_pos_arr = _node_pos_array(
        mapper_plot.graph,
        mapper_plot.dim,
        node_pos,
    )
    edge_pos_arr = _edge_pos_array(
        mapper_plot.graph,
        mapper_plot.dim,
        node_pos,
    )
    _edges_tr = _edges_trace(mapper_plot, edge_pos_arr, node_col, cmap)
    _nodes_tr = _nodes_trace(mapper_plot, node_pos_arr, node_col, title, cmap)
    _layout_ = _layout(width, height)
    return go.Figure(
        data=[_edges_tr, _nodes_tr],
        layout=_layout_)


def _nodes_trace(mapper_plot, node_pos_arr, node_col, title, cmap):
    attr_size = nx.get_node_attributes(mapper_plot.graph, ATTR_SIZE)
    max_size = max(attr_size.values(), default=1.0)
    scatter_text = _text(mapper_plot, node_col)
    marker_size = [25.0 * math.sqrt(attr_size[n] / max_size) for n in
                   mapper_plot.graph.nodes()]
    colors = list(node_col.values())
    scatter = dict(
        name='nodes_trace',
        x=node_pos_arr[0],
        y=node_pos_arr[1],
        mode='markers',
        hoverinfo='text',
        opacity=_NODE_OPACITY,
        text=scatter_text,
        marker=dict(
            showscale=True,
            reversescale=False,
            size=marker_size,
            opacity=_NODE_OPACITY,
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
            line_colorscale=cmap,
            color=colors,
            colorscale=cmap,
            cmin=min(colors, default=None),
            cmax=max(colors, default=None),
            colorbar=_colorbar(mapper_plot, title),
        ),
    )
    if mapper_plot.dim == 3:
        scatter.update(dict(z=node_pos_arr[2]))
        return go.Scatter3d(scatter)
    elif mapper_plot.dim == 2:
        return go.Scatter(scatter)


def _edges_trace(mapper_plot, edge_pos_arr, node_col, cmap):
    scatter = dict(
        name='edges_trace',
        x=edge_pos_arr[0],
        y=edge_pos_arr[1],
        mode='lines',
        opacity=_EDGE_OPACITY,
        line_width=_EDGE_WIDTH,
        line_color=_EDGE_COLOR,
        hoverinfo='skip',
    )
    if mapper_plot.dim == 3:
        colors_avg = []
        for e in mapper_plot.graph.edges():
            c0, c1 = node_col[e[0]], node_col[e[1]]
            colors_avg.append(c0)
            colors_avg.append(c1)
            colors_avg.append(c1)
        colors = list(node_col.values())
        scatter.update(dict(
                z=edge_pos_arr[2],
                line_color=colors_avg,
                line_cmin=min(colors, default=None),
                line_cmax=max(colors, default=None),
                line_colorscale=cmap,
            ),
        )
        return go.Scatter3d(scatter)
    elif mapper_plot.dim == 2:
        scatter.update(dict(
                marker_colorscale=cmap,
                marker_line_colorscale=cmap,
            ),
        )
        return go.Scatter(scatter)


def _layout(width, height):
    line_col = 'rgba(230, 230, 230, 1.0)'
    axis = dict(
        showline=True,
        linewidth=1,
        mirror=True,
        visible=True,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title='',
    )
    scene_axis = dict(
        showgrid=True,
        visible=True,
        backgroundcolor='rgba(0, 0, 0, 0)',
        showaxeslabels=False,
        showline=True,
        linecolor=line_col,
        zerolinecolor=line_col,
        gridcolor=line_col,
        linewidth=1,
        mirror=True,
        showticklabels=False,
        title='',
    )
    return go.Layout(
        uirevision='constant',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        autosize=False,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=10, l=10, r=10, t=10),
        xaxis=axis,
        yaxis=axis,
        width=width,
        height=height,
        scene=dict(
            xaxis=scene_axis,
            yaxis=scene_axis,
            zaxis=scene_axis,
        ),
    )


def _colorbar(mapper_plot, title):
    cbar = dict(
        showticklabels=True,
        outlinewidth=1,
        borderwidth=0,
        orientation='v',
        thickness=0.025,
        thicknessmode='fraction',
        xanchor='left',
        titleside='right',
        ypad=0,
        xpad=0,
        tickwidth=1,
        tickformat='.2g',
        nticks=_TICKS_NUM,
        tickmode='auto',
    )
    if title is not None:
        cbar['title'] = title
    if mapper_plot.dim == 3:
        return go.scatter3d.marker.ColorBar(cbar)
    elif mapper_plot.dim == 2:
        return go.scatter.marker.ColorBar(cbar)


def _text(mapper_plot, colors):
    attr_size = nx.get_node_attributes(mapper_plot.graph, ATTR_SIZE)

    def _lbl(n):
        col = _fmt(colors[n], 3)
        size = _fmt(attr_size[n], 5)
        return f'color: {col}<br>node: {n}<br>size: {size}'
    return [_lbl(n) for n in mapper_plot.graph.nodes()]


def _fmt(x, max_len=3):
    fmt = f'.{max_len}g'
    return f'{x:{fmt}}'
