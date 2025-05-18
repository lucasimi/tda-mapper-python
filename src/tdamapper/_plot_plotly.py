"""
This module provides functionalities to visualize the Mapper graph based on
plotly.
"""

import math
from typing import Optional

import networkx as nx
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from tdamapper.core import ATTR_SIZE, aggregate_graph

_NODE_OUTER_WIDTH = 0.75

_NODE_OUTER_COLOR = "#777"

_NODE_OPACITY = 1.0

_EDGE_WIDTH = 0.75

_EDGE_OPACITY = 1.0

_EDGE_COLOR = "#777"

_TICKS_NUM = 10

NODE_SIZE = 1

CMAP = "Jet"

TITLE = ""


def plot_plotly(
    mapper_plot,
    width: int,
    height: int,
    title: str = TITLE,
    node_size: int = NODE_SIZE,
    colors=None,
    agg=np.nanmean,
    cmap: str = CMAP,
) -> go.Figure:
    # node_col = aggregate_graph(colors, mapper_plot.graph, agg)
    fig = _figure(mapper_plot, width, height, title, node_size, colors, agg, cmap)
    _add_ui_to_layout(mapper_plot, fig, colors, node_size, agg, cmap)
    return fig


def plot_plotly_update(
    mapper_plot,
    fig: go.Figure,
    width: Optional[int] = None,
    height: Optional[int] = None,
    title: Optional[str] = None,
    node_size: Optional[int] = None,
    colors=None,
    agg=None,
    cmap: Optional[str] = None,
) -> go.Figure:
    if (width is not None) and (height is not None):
        _update_layout(fig, width, height)
    if title is not None:
        _set_title(mapper_plot, fig, title)
    if node_size is not None:
        _set_node_size(mapper_plot, fig, node_size)
    if (colors is not None) and (agg is not None):
        _set_colors(mapper_plot, fig, colors, agg)
    if cmap is not None:
        _set_cmap(mapper_plot, fig, cmap)
    # _add_ui_to_layout(mapper_plot, fig, colors, node_size, agg, cmap)
    # TODO: understand how to update this
    return fig


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


def _marker_size(mapper_plot, node_size):
    attr_size = nx.get_node_attributes(mapper_plot.graph, ATTR_SIZE)
    max_size = max(attr_size.values(), default=1.0)
    scale = node_size * (25.0 if mapper_plot.dim == 2 else 15.0)
    marker_size = [
        scale * math.sqrt(attr_size[n] / max_size) for n in mapper_plot.graph.nodes()
    ]
    return marker_size


def _set_cmap(mapper_plot, fig, cmap):
    fig.update_traces(
        patch=dict(
            marker_colorscale=cmap,
            marker_line_colorscale=cmap,
        ),
        selector=dict(name="nodes_trace"),
    )

    if mapper_plot.dim == 3:
        fig.update_traces(
            patch=dict(line_colorscale=cmap),
            selector=dict(name="edges_trace"),
        )
    elif mapper_plot.dim == 2:
        fig.update_traces(
            patch=dict(
                marker_colorscale=cmap,
                marker_line_colorscale=cmap,
            ),
            selector=dict(name="edges_trace"),
        )


def _set_colors(mapper_plot, fig, colors, agg):
    node_col = aggregate_graph(colors, mapper_plot.graph, agg)
    scatter_text = _text(mapper_plot, node_col)
    colors_arr = list(node_col.values())
    fig.update_traces(
        patch=dict(
            text=scatter_text,
            marker=dict(
                color=colors_arr,
                cmin=min(colors_arr, default=None),
                cmax=max(colors_arr, default=None),
            ),
        ),
        selector=dict(name="nodes_trace"),
    )
    if mapper_plot.dim == 3:
        colors_avg = []
        for e in mapper_plot.graph.edges():
            c0, c1 = node_col[e[0]], node_col[e[1]]
            colors_avg.append(c0)
            colors_avg.append(c1)
            colors_avg.append(c1)
        colors_edges_arr = list(node_col.values())
        fig.update_traces(
            patch=dict(
                text=scatter_text,
                marker=dict(
                    line_color=colors_edges_arr,
                    line_cmin=min(colors_edges_arr, default=None),
                    line_cmax=max(colors_edges_arr, default=None),
                ),
            ),
            selector=dict(name="edges_trace"),
        )


def _set_title(mapper_plot, fig, title):
    fig.update_traces(
        patch=dict(
            marker_colorbar=_colorbar(mapper_plot, title),
        ),
        selector=dict(name="nodes_trace"),
    )


def _set_node_size(mapper_plot, fig, node_size):
    fig.update_traces(
        patch=dict(
            marker_size=_marker_size(mapper_plot, node_size),
        ),
        selector=dict(name="nodes_trace"),
    )


def _update_layout(fig, width, height):
    fig.update_layout(
        width=width,
        height=height,
    )


def _figure(mapper_plot, width, height, title, node_size, colors, agg, cmap):
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
    _edges_tr = _edges_trace(mapper_plot, edge_pos_arr)
    _nodes_tr = _nodes_trace(mapper_plot, node_pos_arr)
    _layout_ = _layout(width, height)
    fig = go.Figure(data=[_edges_tr, _nodes_tr], layout=_layout_)

    _set_cmap(mapper_plot, fig, cmap)
    _set_colors(mapper_plot, fig, colors, agg)
    _set_node_size(mapper_plot, fig, node_size)
    _set_title(mapper_plot, fig, title)

    return fig


def _nodes_trace(mapper_plot, node_pos_arr):
    scatter = dict(
        name="nodes_trace",
        x=node_pos_arr[0],
        y=node_pos_arr[1],
        mode="markers",
        hoverinfo="text",
        opacity=_NODE_OPACITY,
        marker=dict(
            showscale=True,
            reversescale=False,
            size=_marker_size(mapper_plot, NODE_SIZE),
            opacity=_NODE_OPACITY,
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
            line_colorscale=CMAP,
            colorscale=CMAP,
            colorbar=_colorbar(mapper_plot, TITLE),
        ),
    )
    if mapper_plot.dim == 3:
        scatter.update(dict(z=node_pos_arr[2]))
        return go.Scatter3d(scatter)
    elif mapper_plot.dim == 2:
        return go.Scatter(scatter)


def _edges_trace(mapper_plot, edge_pos_arr):
    scatter = dict(
        name="edges_trace",
        x=edge_pos_arr[0],
        y=edge_pos_arr[1],
        mode="lines",
        opacity=_EDGE_OPACITY,
        line_width=_EDGE_WIDTH,
        line_color=_EDGE_COLOR,
        hoverinfo="skip",
    )
    if mapper_plot.dim == 3:
        scatter.update(
            dict(
                z=edge_pos_arr[2],
                line_colorscale=CMAP,
            ),
        )
        return go.Scatter3d(scatter)
    elif mapper_plot.dim == 2:
        scatter.update(
            dict(
                marker_colorscale=CMAP,
                marker_line_colorscale=CMAP,
            ),
        )
        return go.Scatter(scatter)


def _colorbar(mapper_plot, title):
    cbar = dict(
        showticklabels=True,
        outlinewidth=1,
        borderwidth=0,
        orientation="v",
        thickness=0.025,
        thicknessmode="fraction",
        xanchor="left",
        title_side="right",
        ypad=0,
        xpad=0,
        tickwidth=1,
        tickformat=".2g",
        nticks=_TICKS_NUM,
        tickmode="auto",
    )
    if title is not None:
        cbar["title"] = title
    if mapper_plot.dim == 3:
        return go.scatter3d.marker.ColorBar(cbar)
    elif mapper_plot.dim == 2:
        return go.scatter.marker.ColorBar(cbar)


def _text(mapper_plot, colors):
    attr_size = nx.get_node_attributes(mapper_plot.graph, ATTR_SIZE)

    def _lbl(n):
        col = _fmt(colors[n], 3)
        size = _fmt(attr_size[n], 5)
        return f"color: {col}<br>node: {n}<br>size: {size}"

    return [_lbl(n) for n in mapper_plot.graph.nodes()]


def _fmt(x, max_len=3):
    fmt = f".{max_len}g"
    return f"{x:{fmt}}"


def _layout(width, height):
    line_col = "rgba(230, 230, 230, 1.0)"
    axis = dict(
        showline=True,
        linewidth=1,
        mirror=True,
        visible=True,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        automargin=False,
        title="",
    )
    scene_axis = dict(
        showgrid=True,
        visible=True,
        backgroundcolor="rgba(0, 0, 0, 0)",
        showaxeslabels=False,
        showline=True,
        linecolor=line_col,
        zerolinecolor=line_col,
        gridcolor=line_col,
        linewidth=1,
        mirror=True,
        showticklabels=False,
        title="",
    )
    return go.Layout(
        plot_bgcolor="rgba(0, 0, 0, 0)",
        autosize=False,
        showlegend=False,
        hovermode="closest",
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


def _get_plotly_js_colorscales():
    """
    Return a sorted list of all case-sensitive color-scale names
    supported by Plotly.js (via the Python plotly.colors wrapper).
    Includes sequential, diverging, and cyclical scales.
    """
    modules = ["sequential", "diverging", "cyclical"]
    all_scales = []
    for mod in modules:
        m = getattr(pc, mod)
        # keep only public scale names (start with uppercase letter)
        valid = [name for name in dir(m) if name[0].isupper()]
        all_scales.extend(valid)
    # dedupe and sort
    return sorted(set(all_scales))


def _sanitize_cmap(cmap):
    cmaps = _get_plotly_js_colorscales()
    cmaps_dict = {c.lower(): c for c in cmaps}
    return cmaps_dict.get(cmap.lower(), None)


def _add_ui_to_layout(mapper_plot, mapper_fig, colors, node_size, agg, cmap):
    cmaps_raw = [
        cmap,
        "Jet",
        "Viridis",
        "Cividis",
        "Spectral",
        "Portland",
        "Blues",
        "Greens",
        "HSV",
        "Twilight",
    ]
    cmaps = [_sanitize_cmap(c) for c in cmaps_raw]
    menu_color = _ui_color(mapper_plot, colors, agg)
    menu_cmap = _ui_cmap(mapper_plot, cmaps)
    slider_size = _ui_node_size(mapper_plot, node_size)
    mapper_fig.update_layout(
        updatemenus=[menu_cmap, menu_color],
        sliders=[slider_size],
    )


def _ui_cmap(mapper_plot, cmaps):
    target_traces = [1] if mapper_plot.dim == 2 else [0, 1]

    def _update_cmap(cmap):
        if mapper_plot.dim == 2:
            return {
                "marker.colorscale": [cmap],
                "marker.line.colorscale": [cmap],
            }
        elif mapper_plot.dim == 3:
            return {
                "marker.colorscale": [None, cmap],
                "marker.line.colorscale": [None, cmap],
                "line.colorscale": [cmap, None],
            }

    return dict(
        buttons=[
            dict(
                label=cmap,
                method="restyle",
                args=[_update_cmap(cmap), target_traces],
            )
            for cmap in cmaps
        ],
        x=0.25,
        xanchor="left",
        y=1.0,
        yanchor="top",
        direction="down",
    )


def _ui_node_size(mapper_plot, node_size):
    steps = [
        dict(
            method="restyle",
            label=f"{size}",
            args=[
                {"marker.size": [_marker_size(mapper_plot, size)]},
                [1],
            ],
        )
        for size in [node_size * x / 10.0 for x in range(1, 20)]
    ]

    return dict(
        active=len(steps) // 2,
        currentvalue={"prefix": "Node size: "},
        steps=steps,
        x=0.0,
        y=0.0,
        xanchor="left",
        len=0.3,
        yanchor="bottom",
    )


def _ui_color(mapper_plot, colors, agg):
    colors_arr = np.array(colors)
    colors_num = colors_arr.shape[1] if colors_arr.ndim == 2 else 1

    def _colors_agg(i):
        if i is None:
            arr = colors_arr
        else:
            arr = colors_arr[:, i] if colors_arr.ndim == 2 else colors_arr
        return aggregate_graph(arr, mapper_plot.graph, agg)

    def _colors(i):
        return list(_colors_agg(i).values())

    def _edge_colors(i):
        colors_avg = []
        colors_agg = _colors_agg(i)
        for edge in mapper_plot.graph.edges():
            c0, c1 = colors_agg[edge[0]], colors_agg[edge[1]]
            colors_avg.append(c0)
            colors_avg.append(c1)
            colors_avg.append(c1)
        return colors_avg

    def _update_colors(i):
        arr = _colors(i)
        if mapper_plot.dim == 2:
            return {
                "marker.color": [arr],
                "marker.cmax": [max(arr, default=None)],
                "marker.cmin": [min(arr, default=None)],
            }
        elif mapper_plot.dim == 3:
            arr_edge = _edge_colors(i)
            return {
                "marker.color": [None, arr],
                "marker.cmax": [None, max(arr, default=None)],
                "marker.cmin": [None, min(arr, default=None)],
                "line.color": [arr_edge, None],
                "line.cmax": [max(arr_edge, default=None), None],
                "line.cmin": [min(arr_edge, default=None), None],
            }

    target_traces = [1] if mapper_plot.dim == 2 else [0, 1]

    buttons = [
        dict(
            label="Feature: [All]",
            method="restyle",
            args=[_update_colors(None), target_traces],
        )
    ]

    if colors_num > 1:
        buttons.extend(
            [
                dict(
                    label=f"Feature {i}",
                    method="restyle",
                    args=[_update_colors(i), target_traces],
                )
                for i in range(colors_num)
            ]
        )

    return dict(
        buttons=buttons,
        x=0.0,
        xanchor="left",
        y=1.0,
        yanchor="top",
        direction="down",
    )
