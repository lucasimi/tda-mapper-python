"""
This module provides functionalities to visualize the Mapper graph based on
plotly.
"""

import math
from typing import List, Optional, Union

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

_NODES_TRACE = "nodes_trace"

_EDGES_TRACE = "edges_trace"

DEFAULT_NODE_SIZE = 1

DEFAULT_CMAP = "Jet"

DEFAULT_TITLE = ""


def _get_plotly_colorscales():
    modules = ["sequential", "diverging", "cyclical"]
    all_scales = []
    for mod in modules:
        m = getattr(pc, mod)
        # keep only public scale names (start with uppercase letter)
        valid = [name for name in dir(m) if name[0].isupper()]
        all_scales.extend(valid)
    cmaps = sorted(set(all_scales))
    return {c.lower(): c for c in cmaps}


PLOTLY_CMAPS = _get_plotly_colorscales()


class PlotlyUI:

    def __init__(self):
        self.menu_cmap = None
        self.menu_color = None
        self.slider_size = None

    def set_menu_cmap(self, mapper_plot, cmaps: Optional[List[str]]) -> None:
        if cmaps is None:
            return
        cmaps_plotly = [PLOTLY_CMAPS.get(c.lower()) for c in cmaps]
        self.menu_cmap = _ui_cmap(mapper_plot, cmaps_plotly)

    def set_menu_color(self, mapper_plot, colors, titles, agg):
        self.menu_color = _ui_color(mapper_plot, colors, titles, agg)

    def set_slider_size(self, mapper_plot, node_sizes):
        self.slider_size = _ui_node_size(mapper_plot, node_sizes)


def _to_cmaps(cmap: Optional[Union[str, List[str]]]) -> List[str]:
    """Convert a single cmap or a list of cmaps to a list of cmaps."""
    if cmap is None:
        return [DEFAULT_CMAP]
    if isinstance(cmap, str):
        return [cmap]
    elif isinstance(cmap, list):
        return cmap
    else:
        raise ValueError(f"Invalid cmap type: {type(cmap)}. Expected str or list[str].")


def _to_colors(colors: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Convert colors to a numpy array."""
    colors_arr = np.array(colors)
    if colors_arr.ndim == 1:
        return colors_arr.reshape(-1, 1)
    elif colors_arr.ndim == 2:
        return colors_arr
    else:
        raise ValueError(
            f"Invalid colors shape: {colors_arr.shape}. Expected 1D or 2D array."
        )


def _to_titles(title: Optional[Union[str, List[str]]], colors_num: int) -> List[str]:
    if title is None:
        return [f"{i}" for i in range(colors_num)]
    elif isinstance(title, str):
        return [f"{title} {i}" for i in range(colors_num)]
    elif isinstance(title, list) and len(title) == colors_num:
        return title
    else:
        raise ValueError(
            f"Invalid title type: {type(title)}. Expected str or list[str]."
        )


def _to_node_sizes(
    node_size: Optional[Union[int, float, List[Union[int, float]]]]
) -> List[float]:
    if isinstance(node_size, (int, float)):
        return [node_size]
    elif isinstance(node_size, list):
        return node_size
    else:
        raise ValueError(
            f"Invalid node_size type: {type(node_size)}. "
            "Expected int, float or list[int, float]."
        )


def plot_plotly(
    mapper_plot,
    width: int,
    height: int,
    colors: Union[np.ndarray, List[float]],
    node_size: Optional[Union[int, float, List[Union[int, float]]]] = None,
    title: Optional[Union[str, List[str]]] = None,
    agg=np.nanmean,
    cmap: Optional[Union[str, List[str]]] = None,
) -> go.Figure:
    cmaps = _to_cmaps(cmap)
    colors = _to_colors(colors)
    colors_num = colors.shape[1]
    titles = _to_titles(title, colors_num)
    node_sizes = _to_node_sizes(node_size)
    fig = _figure(mapper_plot, width, height, node_sizes, colors, titles, agg, cmaps)
    ui = PlotlyUI()
    ui.set_menu_cmap(mapper_plot, cmaps)
    ui.set_menu_color(mapper_plot, colors, titles, agg)
    ui.set_slider_size(mapper_plot, node_sizes)
    _set_ui(fig, ui)
    return fig


def plot_plotly_update(
    mapper_plot,
    fig: go.Figure,
    width: Optional[int] = None,
    height: Optional[int] = None,
    node_size: Optional[Union[int, float, List[Union[int, float]]]] = None,
    colors=None,
    title: Optional[Union[str, List[str]]] = None,
    agg=None,
    cmap: Optional[Union[str, List[str]]] = None,
) -> go.Figure:
    ui = PlotlyUI()
    cmaps = None
    if cmap is not None:
        cmaps = _to_cmaps(cmap)
        ui.set_menu_cmap(mapper_plot, cmaps)
    colors_num = 0
    if colors is not None:
        colors = _to_colors(colors)
        colors_num = colors.shape[1]
    titles = None
    if title is not None:
        titles = _to_titles(title, colors_num)
    if titles is not None and colors is not None and agg is not None:
        ui.set_menu_color(mapper_plot, colors, titles, agg)
    node_sizes = None
    if node_size is not None:
        node_sizes = _to_node_sizes(node_size)
        ui.set_slider_size(mapper_plot, node_sizes)
    _update(
        mapper_plot,
        fig,
        width=width,
        height=height,
        titles=titles,
        node_sizes=node_sizes,
        colors=colors,
        agg=agg,
        cmaps=cmaps,
    )
    _set_ui(fig, ui)
    return fig


def _node_pos_array(graph: nx.Graph, dim: int, node_pos):
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


def _marker_size(mapper_plot, node_size: float) -> List[float]:
    attr_size = nx.get_node_attributes(mapper_plot.graph, ATTR_SIZE)
    max_size = max(attr_size.values(), default=1.0)
    scale = node_size * (25.0 if mapper_plot.dim == 2 else 15.0)
    marker_size = [
        scale * math.sqrt(attr_size[n] / max_size) for n in mapper_plot.graph.nodes()
    ]
    return marker_size


def _get_cmap_rgb(cmap: str):
    """Return a colorscale in [[float, 'rgb(r,g,b)']] format."""
    base_scale = pc.get_colorscale(cmap)
    # If it's already in [float, color] format, we're good
    return [[pos, color] for pos, color in base_scale]


def _set_cmap(mapper_plot, fig: go.Figure, cmap: str) -> None:
    cmap_rgb = _get_cmap_rgb(cmap)
    fig.update_traces(
        patch=dict(
            marker_colorscale=cmap_rgb,
            marker_line_colorscale=cmap_rgb,
        ),
        selector=dict(name=_NODES_TRACE),
    )

    if mapper_plot.dim == 3:
        fig.update_traces(
            patch=dict(line_colorscale=cmap_rgb),
            selector=dict(name=_EDGES_TRACE),
        )
    elif mapper_plot.dim == 2:
        fig.update_traces(
            patch=dict(
                marker_colorscale=cmap_rgb,
                marker_line_colorscale=cmap_rgb,
            ),
            selector=dict(name=_EDGES_TRACE),
        )


def _set_colors(mapper_plot, fig: go.Figure, colors, agg):
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
        selector=dict(name=_NODES_TRACE),
    )
    if mapper_plot.dim == 3:
        colors_avg = []
        for e in mapper_plot.graph.edges():
            c0, c1 = node_col[e[0]], node_col[e[1]]
            colors_avg.append(c0)
            colors_avg.append(c1)
            colors_avg.append(c1)
        fig.update_traces(
            patch=dict(
                marker=dict(
                    line_color=colors_avg,
                    line_cmin=min(colors_arr, default=None),
                    line_cmax=max(colors_arr, default=None),
                ),
            ),
            selector=dict(name=_EDGES_TRACE),
        )


def _set_title(fig: go.Figure, color_name: str):
    fig.update_traces(
        patch=dict(
            marker_colorbar=_colorbar(color_name),
        ),
        selector=dict(name=_NODES_TRACE),
    )


def _set_node_size(mapper_plot, fig: go.Figure, node_size: float) -> None:
    fig.update_traces(
        patch=dict(
            marker_size=_marker_size(mapper_plot, node_size),
        ),
        selector=dict(name=_NODES_TRACE),
    )


def _set_width(fig: go.Figure, width: int) -> None:
    fig.update_layout(
        width=width,
    )


def _set_height(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height,
    )


def _figure(
    mapper_plot,
    width: int,
    height: int,
    node_sizes: List[float],
    colors: np.ndarray,
    titles: List[str],
    agg,
    cmaps: List[str],
) -> go.Figure:
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
    _layout_ = _layout()
    fig = go.Figure(data=[_edges_tr, _nodes_tr], layout=_layout_)

    _update(
        mapper_plot,
        fig,
        width=width,
        height=height,
        titles=titles,
        node_sizes=node_sizes,
        colors=colors,
        agg=agg,
        cmaps=cmaps,
    )

    return fig


def _update(
    mapper_plot,
    fig: go.Figure,
    width: Optional[int] = None,
    height: Optional[int] = None,
    titles: Optional[List[str]] = None,
    node_sizes: Optional[List[float]] = None,
    colors=None,
    agg=None,
    cmaps: Optional[List[str]] = None,
) -> go.Figure:
    if width is not None:
        _set_width(fig, width)
    if height is not None:
        _set_height(fig, height)
    if titles is not None:
        _set_title(fig, titles[0])
    if node_sizes is not None:
        _set_node_size(mapper_plot, fig, node_sizes[len(node_sizes) // 2])
    if (colors is not None) and (agg is not None):
        _set_colors(mapper_plot, fig, colors[:, 0], agg)
    if cmaps is not None:
        _set_cmap(mapper_plot, fig, cmaps[0])
    return fig


def _nodes_trace(mapper_plot, node_pos_arr):
    scatter = dict(
        name=_NODES_TRACE,
        x=node_pos_arr[0],
        y=node_pos_arr[1],
        mode="markers",
        hoverinfo="text",
        opacity=_NODE_OPACITY,
        marker=dict(
            showscale=True,
            reversescale=False,
            size=_marker_size(mapper_plot, DEFAULT_NODE_SIZE),
            opacity=_NODE_OPACITY,
            line_width=_NODE_OUTER_WIDTH,
            line_color=_NODE_OUTER_COLOR,
            line_colorscale=DEFAULT_CMAP,
            colorscale=DEFAULT_CMAP,
            colorbar=_colorbar(DEFAULT_TITLE),
        ),
    )
    if mapper_plot.dim == 3:
        scatter.update(dict(z=node_pos_arr[2]))
        return go.Scatter3d(scatter)
    elif mapper_plot.dim == 2:
        return go.Scatter(scatter)


def _edges_trace(mapper_plot, edge_pos_arr):
    scatter = dict(
        name=_EDGES_TRACE,
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
                line_colorscale=DEFAULT_CMAP,
            ),
        )
        return go.Scatter3d(scatter)
    elif mapper_plot.dim == 2:
        scatter.update(
            dict(
                marker_colorscale=DEFAULT_CMAP,
                marker_line_colorscale=DEFAULT_CMAP,
            ),
        )
        return go.Scatter(scatter)


def _colorbar(title: str) -> dict:
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
    return cbar


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


def _layout() -> go.Layout:
    line_col = "rgba(230, 230, 230, 1.0)"
    axis = dict(
        showline=False,
        linewidth=1,
        mirror=True,
        visible=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        automargin=False,
        title="",
    )
    scene_axis = dict(
        showgrid=False,
        visible=False,
        backgroundcolor="rgba(0, 0, 0, 0)",
        showaxeslabels=False,
        showline=False,
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
        scene=dict(
            xaxis=scene_axis,
            yaxis=scene_axis,
            zaxis=scene_axis,
        ),
    )


def _set_ui(mapper_fig: go.Figure, plotly_ui: PlotlyUI) -> None:
    menus = []
    sliders = []
    x = 0.0
    if plotly_ui.menu_cmap:
        plotly_ui.menu_cmap["x"] = x
        x += 0.25
        menus.append(plotly_ui.menu_cmap)
    if plotly_ui.menu_color:
        plotly_ui.menu_color["x"] = x
        menus.append(plotly_ui.menu_color)
    if plotly_ui.slider_size:
        plotly_ui.slider_size["x"] = 0.0
        sliders.append(plotly_ui.slider_size)
    mapper_fig.update_layout(
        updatemenus=menus,
        sliders=sliders,
    )


def _ui_cmap(mapper_plot, cmaps: List[str]) -> dict:
    target_traces = [1] if mapper_plot.dim == 2 else [0, 1]

    def _update_cmap(cmap: str) -> dict:
        cmap_rgb = _get_cmap_rgb(cmap)
        if mapper_plot.dim == 2:
            return {
                "marker.colorscale": [cmap_rgb],
                "marker.line.colorscale": [cmap_rgb],
            }
        elif mapper_plot.dim == 3:
            return {
                "marker.colorscale": [None, cmap_rgb],
                "marker.line.colorscale": [None, cmap_rgb],
                "line.colorscale": [cmap_rgb, None],
            }
        return {}

    buttons = []
    if len(cmaps) > 1:
        buttons = [
            dict(
                label=cmap,
                method="restyle",
                args=[_update_cmap(cmap), target_traces],
            )
            for cmap in cmaps
        ]

    return dict(
        buttons=buttons,
        x=0.25,
        xanchor="left",
        y=1.0,
        yanchor="top",
        direction="down",
    )


def _ui_node_size(mapper_plot, node_sizes: List[float]) -> dict:
    steps = [
        dict(
            method="restyle",
            label=f"{size}",
            args=[
                {"marker.size": [_marker_size(mapper_plot, size)]},
                [1],
            ],
        )
        for size in node_sizes
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


def _ui_color(mapper_plot, colors, titles: List[str], agg) -> dict:
    colors_arr = np.array(colors)
    colors_num = colors_arr.shape[1] if colors_arr.ndim == 2 else 1

    def _colors_agg(i: int) -> dict:
        if i is None:
            arr = colors_arr
        else:
            arr = colors_arr[:, i] if colors_arr.ndim == 2 else colors_arr
        return aggregate_graph(arr, mapper_plot.graph, agg)

    def _colors(i: int) -> List[float]:
        return list(_colors_agg(i).values())

    def _edge_colors(i: int) -> List[float]:
        colors_avg = []
        colors_agg = _colors_agg(i)
        for edge in mapper_plot.graph.edges():
            c0, c1 = colors_agg[edge[0]], colors_agg[edge[1]]
            colors_avg.append(c0)
            colors_avg.append(c1)
            colors_avg.append(c1)
        return colors_avg

    def _update_colors(i: int) -> dict:
        arr_agg = _colors_agg(i)
        arr = list(arr_agg.values())
        scatter_text = _text(mapper_plot, arr_agg)
        cbar = _colorbar(titles[i])
        if mapper_plot.dim == 2:
            return {
                "text": [scatter_text],
                "marker.colorbar": [cbar],
                "marker.color": [arr],
                "marker.cmax": [max(arr, default=None)],
                "marker.cmin": [min(arr, default=None)],
            }
        elif mapper_plot.dim == 3:
            arr_edge = _edge_colors(i)
            return {
                "text": [None, scatter_text],
                "marker.colorbar": [None, cbar],
                "marker.color": [None, arr],
                "marker.cmax": [None, max(arr, default=None)],
                "marker.cmin": [None, min(arr, default=None)],
                "line.color": [arr_edge, None],
                "line.cmax": [max(arr_edge, default=None), None],
                "line.cmin": [min(arr_edge, default=None), None],
            }
        return {}

    target_traces = [1] if mapper_plot.dim == 2 else [0, 1]

    buttons = []
    if colors.shape[1] > 1:
        buttons = [
            dict(
                label=f"{titles[i]}",
                method="restyle",
                args=[_update_colors(i), target_traces],
            )
            for i in range(colors_num)
        ]

    return dict(
        buttons=buttons,
        x=0.0,
        xanchor="left",
        y=1.0,
        yanchor="top",
        direction="down",
    )
