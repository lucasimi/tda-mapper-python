"""
This module provides functionalities to visualize the Mapper graph based on
plotly.
"""

import math
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

from tdamapper.core import ATTR_SIZE, aggregate_graph

_NODE_OUTER_WIDTH = 0.75

_NODE_OUTER_COLOR = "#777"

_NODE_OPACITY = 1.0

_EDGE_WIDTH_2D = 0.75

_EDGE_WIDTH_3D = 1.5

_EDGE_OPACITY = 1.0

_EDGE_COLOR = "#777"

_TICKS_NUM = 10

_NODES_TRACE = "nodes_trace"

_EDGES_TRACE = "edges_trace"

_DEFAULT_SPACING = 0.25

_MAX_SIZEREF = 1000000000

_MARKER_SIZE_FACTOR = 400.0

MENU_DARK_MODE_NAME = "menu_dark_mode"

MENU_CMAP_NAME = "menu_cmap"

MENU_COLOR_NAME = "menu_color"

SLIDER_NODE_SIZE_NAME = "slider_node_size"

FONT_COLOR_LIGHT = "#2a3f5f"

FONT_COLOR_DARK = "#f2f5fa"

DEFAULT_NODE_SIZE = 1

DEFAULT_CMAP = "Jet"

DEFAULT_TITLE = ""


def _get_plotly_colorscales() -> Dict[str, str]:
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


def _fmt(x: Any, max_len: int = 3) -> str:
    fmt = f".{max_len}g"
    return f"{x:{fmt}}"


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
        if colors_num == 1:
            return [title]
        else:
            return [f"{title} [{i}]" for i in range(colors_num)]
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


def _get_cmap_rgb(cmap: str):
    """Return a colorscale in [[float, 'rgb(r,g,b)']] format."""
    base_scale = pc.get_colorscale(cmap)
    # If it's already in [float, color] format, we're good
    return [[pos, color] for pos, color in base_scale]


def plot_plotly(
    mapper_plot,
    colors: Union[np.ndarray, List[float]],
    node_size: Optional[Union[int, float, List[Union[int, float]]]] = None,
    title: Optional[Union[str, List[str]]] = None,
    agg=np.nanmean,
    cmap: Optional[Union[str, List[str]]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    cmaps = _to_cmaps(cmap)
    colors = _to_colors(colors)
    colors_num = colors.shape[1]
    titles = _to_titles(title, colors_num)
    node_sizes = _to_node_sizes(node_size)
    plot = PlotlyPlot(mapper_plot)
    fig = plot.plot(
        colors=colors,
        node_sizes=node_sizes,
        titles=titles,
        agg=agg,
        cmaps=cmaps,
        width=width,
        height=height,
    )
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
    plot = PlotlyPlot(mapper_plot, fig)
    cmaps = None
    if cmap is not None:
        cmaps = _to_cmaps(cmap)

    colors_num = 0
    if colors is not None:
        colors = _to_colors(colors)
        colors_num = colors.shape[1]

    titles = None
    if title is not None:
        titles = _to_titles(title, colors_num)

    node_sizes = None
    if node_size is not None:
        node_sizes = _to_node_sizes(node_size)

    plot.set_ui(
        cmaps=cmaps,
        colors=colors,
        titles=titles,
        agg=agg,
        node_sizes=node_sizes,
    )
    plot.update_figure(
        width=width,
        height=height,
        titles=titles,
        node_sizes=node_sizes,
        colors=colors,
        agg=agg,
        cmaps=cmaps,
    )
    return fig


class PlotlyPlot:

    def __init__(self, mapper_plot, fig: Optional[go.Figure] = None):
        self.mapper_plot = mapper_plot
        self.fig = fig
        self.graph = mapper_plot.graph
        self.positions = mapper_plot.positions
        self.dim = mapper_plot.dim

    def plot(
        self,
        colors: np.ndarray,
        node_sizes: List[float],
        titles: List[str],
        agg,
        cmaps: List[str],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> go.Figure:
        self.set_figure(
            node_sizes=node_sizes,
            colors=colors,
            titles=titles,
            agg=agg,
            cmaps=cmaps,
            width=width,
            height=height,
        )
        self.set_ui(
            cmaps=cmaps,
            colors=colors,
            titles=titles,
            agg=agg,
            node_sizes=node_sizes,
        )
        return self.fig

    def _node_pos_array(self):
        return tuple(
            [self.positions[n][i] for n in self.graph.nodes()] for i in range(self.dim)
        )

    def _edge_pos_array(self):
        edges_arr = tuple([] for i in range(self.dim))
        for edge in self.graph.edges():
            pos0, pos1 = self.positions[edge[0]], self.positions[edge[1]]
            for i in range(self.dim):
                edges_arr[i].append(pos0[i])
                edges_arr[i].append(pos1[i])
                edges_arr[i].append(None)
        return edges_arr

    def _marker_size(self) -> List[float]:
        attr_size = nx.get_node_attributes(self.graph, ATTR_SIZE)
        max_size = max(attr_size.values(), default=1.0)
        marker_size = [
            _MARKER_SIZE_FACTOR * attr_size[n] / max_size for n in self.graph.nodes()
        ]
        return marker_size

    def set_cmap(self, cmap: str) -> None:
        if self.fig is None:
            return
        cmap_rgb = _get_cmap_rgb(cmap)
        self.fig.update_traces(
            patch=dict(
                marker_colorscale=cmap_rgb,
                marker_line_colorscale=cmap_rgb,
            ),
            selector=dict(name=_NODES_TRACE),
        )

        if self.dim == 3:
            self.fig.update_traces(
                patch=dict(line_colorscale=cmap_rgb),
                selector=dict(name=_EDGES_TRACE),
            )
        elif self.dim == 2:
            self.fig.update_traces(
                patch=dict(
                    marker_colorscale=cmap_rgb,
                    marker_line_colorscale=cmap_rgb,
                ),
                selector=dict(name=_EDGES_TRACE),
            )

    def _edge_colors_from_node_colors(
        self,
        node_colors: Dict[Any, float],
    ) -> List[float]:
        edge_col = []
        for e in self.graph.edges():
            c0, c1 = node_colors[e[0]], node_colors[e[1]]
            edge_col.append(c0)
            edge_col.append(c1)
            edge_col.append(c1)
        return edge_col

    def _set_colors(self, colors, agg):
        node_col_agg = aggregate_graph(colors, self.graph, agg)
        node_col_arr = list(node_col_agg.values())
        scatter_text = self._text(node_col_agg)
        self.fig.update_traces(
            patch=dict(
                text=scatter_text,
                marker=dict(
                    opacity=_NODE_OPACITY,
                    color=node_col_arr,
                    cmin=min(node_col_arr, default=None),
                    cmax=max(node_col_arr, default=None),
                ),
            ),
            selector=dict(name=_NODES_TRACE),
        )
        if self.dim == 3:
            edge_col = self._edge_colors_from_node_colors(
                node_col_agg,
            )
            self.fig.update_traces(
                patch=dict(
                    line_color=edge_col,
                    line_cmin=min(node_col_arr, default=None),
                    line_cmax=max(node_col_arr, default=None),
                ),
                selector=dict(name=_EDGES_TRACE),
            )

    def set_title(self, color_name: str) -> None:
        if self.fig is None:
            return
        self.fig.update_traces(
            patch=dict(
                marker_colorbar=self._colorbar(color_name),
            ),
            selector=dict(name=_NODES_TRACE),
        )

    def set_node_size(self, node_size: float) -> None:
        if self.fig is None:
            return
        self.fig.update_traces(
            patch=dict(
                marker_sizeref=(
                    _MAX_SIZEREF if node_size == 0.0 else (1.0 / node_size) ** 2
                ),
                marker_sizemin=1.0,
                marker_sizemode="area",
            ),
            selector=dict(name=_NODES_TRACE),
        )

    def set_width(self, width: int) -> None:
        if self.fig is None:
            return
        self.fig.update_layout(
            width=width,
        )

    def set_height(self, height: int) -> None:
        if self.fig is None:
            return
        self.fig.update_layout(
            height=height,
        )

    def set_figure(
        self,
        node_sizes: List[float],
        colors: np.ndarray,
        titles: List[str],
        agg,
        cmaps: List[str],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        node_pos_arr = self._node_pos_array()
        edge_pos_arr = self._edge_pos_array()
        _edges_tr = self._edges_trace(edge_pos_arr)
        _nodes_tr = self._nodes_trace(node_pos_arr)
        _layout_ = self._layout()
        fig = go.Figure(data=[_edges_tr, _nodes_tr], layout=_layout_)
        self.fig = fig
        self.update_figure(
            titles=titles,
            node_sizes=node_sizes,
            colors=colors,
            agg=agg,
            cmaps=cmaps,
            width=width,
            height=height,
        )

    def update_figure(
        self,
        titles: Optional[List[str]] = None,
        node_sizes: Optional[List[float]] = None,
        colors=None,
        agg=None,
        cmaps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        if width is not None:
            self.set_width(width)
        if height is not None:
            self.set_height(height)
        if titles is not None:
            self.set_title(titles[0])
        if node_sizes is not None:
            self.set_node_size(node_sizes[len(node_sizes) // 2])
        if (colors is not None) and (agg is not None):
            self._set_colors(colors[:, 0], agg)
        if cmaps is not None:
            self.set_cmap(cmaps[0])

    def _nodes_trace(self, node_pos_arr) -> Union[go.Scatter, go.Scatter3d]:
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
                size=self._marker_size(),
                opacity=_NODE_OPACITY,
                line_width=_NODE_OUTER_WIDTH,
                line_color=_NODE_OUTER_COLOR,
                line_colorscale=DEFAULT_CMAP,
                colorscale=DEFAULT_CMAP,
                colorbar=self._colorbar(DEFAULT_TITLE),
            ),
        )
        if self.dim == 3:
            scatter.update(dict(z=node_pos_arr[2]))
            return go.Scatter3d(scatter)
        else:
            return go.Scatter(scatter)

    def _edges_trace(self, edge_pos_arr) -> Union[go.Scatter, go.Scatter3d]:
        scatter = dict(
            name=_EDGES_TRACE,
            x=edge_pos_arr[0],
            y=edge_pos_arr[1],
            mode="lines",
            opacity=_EDGE_OPACITY,
            line_color=_EDGE_COLOR,
            hoverinfo="skip",
        )
        if self.dim == 3:
            scatter.update(
                dict(
                    z=edge_pos_arr[2],
                    line_colorscale=DEFAULT_CMAP,
                    line_width=_EDGE_WIDTH_3D,
                    marker_line_width=_EDGE_WIDTH_3D,
                ),
            )
            return go.Scatter3d(scatter)
        else:
            scatter.update(
                dict(
                    marker_colorscale=DEFAULT_CMAP,
                    marker_line_width=_EDGE_WIDTH_2D,
                    marker_line_colorscale=DEFAULT_CMAP,
                    line_width=_EDGE_WIDTH_2D,
                ),
            )
            return go.Scatter(scatter)

    def _colorbar(self, title: str) -> dict:
        cbar = dict(
            orientation="v",
            showticklabels=True,
            outlinewidth=1,
            borderwidth=0,
            thicknessmode="fraction",
            title_side="right",
            title_text=title,
            xanchor="right",
            x=1.0,
            ypad=0,
            xpad=0,
            tickwidth=1,
            tickformat=".2g",
            nticks=_TICKS_NUM,
            tickmode="auto",
        )
        return cbar

    def _text(self, colors: Dict[Any, Any]) -> List[str]:
        attr_size = nx.get_node_attributes(self.graph, ATTR_SIZE)

        def _lbl(n):
            col = _fmt(colors[n], 3)
            size = _fmt(attr_size[n], 5)
            return f"color: {col}<br>node: {n}<br>size: {size}"

        return [_lbl(n) for n in self.graph.nodes()]

    def _layout(self) -> go.Layout:
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
            linewidth=1,
            mirror=True,
            showticklabels=False,
            title="",
        )
        return go.Layout(
            autosize=True,
            height=None,
            width=None,
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
            font_color=FONT_COLOR_LIGHT,
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

    def set_ui(
        self,
        cmaps: Optional[List[str]],
        colors: Optional[np.ndarray],
        titles: Optional[List[str]],
        agg,
        node_sizes: Optional[List[float]],
    ) -> None:
        if self.fig is None:
            return

        ui_menu_dark_mode = None
        ui_menu_cmap = None
        ui_menu_color = None
        ui_slider_size = None
        for m in self.fig.layout.updatemenus:
            if m.name == MENU_DARK_MODE_NAME:
                ui_menu_dark_mode = m
            elif m.name == MENU_CMAP_NAME:
                ui_menu_cmap = m
            elif m.name == MENU_COLOR_NAME:
                ui_menu_color = m
        for s in self.fig.layout.sliders:
            if s.name == SLIDER_NODE_SIZE_NAME:
                ui_slider_size = s

        ui_menu_dark_mode = self.ui_menu_dark_mode()

        if cmaps is not None:
            cmaps_plotly = [PLOTLY_CMAPS.get(c.lower()) for c in cmaps]
            ui_menu_cmap = self._ui_menu_cmap(cmaps_plotly)

        if colors is not None and agg is not None and titles is not None:
            ui_menu_color = self._ui_menu_color(colors, titles, agg)

        if node_sizes is not None:
            ui_slider_size = self._ui_slider_node_size(node_sizes)

        menus = [m for m in [ui_menu_dark_mode, ui_menu_cmap, ui_menu_color] if m]
        sliders = [s for s in [ui_slider_size] if s]

        self.fig.layout.updatemenus = ()
        self.fig.layout.sliders = ()

        self.fig.update_layout(
            updatemenus=menus,
            sliders=sliders,
        )

    def ui_menu_dark_mode(self) -> Dict:
        buttons = [
            dict(
                label="Light",
                method="relayout",
                args=[
                    {
                        "font.color": FONT_COLOR_LIGHT,
                        "paper_bgcolor": "white",
                        "plot_bgcolor": "white",
                    }
                ],
            ),
            dict(
                label="Dark",
                method="relayout",
                args=[
                    {
                        "font.color": FONT_COLOR_DARK,
                        "paper_bgcolor": "black",
                        "plot_bgcolor": "black",
                    }
                ],
            ),
        ]
        return dict(
            name=MENU_DARK_MODE_NAME,
            buttons=buttons,
            direction="down",
            active=0,
            x=0.25,
            y=1.0,
            xanchor="center",
            yanchor="top",
        )

    def _ui_menu_cmap(self, cmaps: List[str]) -> dict:
        target_traces = [1] if self.dim == 2 else [0, 1]

        def _update_cmap(cmap: str) -> dict:
            cmap_rgb = _get_cmap_rgb(cmap)
            if self.dim == 2:
                return {
                    "marker.colorscale": [cmap_rgb],
                    "marker.line.colorscale": [cmap_rgb],
                }
            elif self.dim == 3:
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
            name=MENU_CMAP_NAME,
            buttons=buttons,
            direction="down",
            x=0.5,
            y=1.0,
            xanchor="center",
            yanchor="top",
        )

    def _ui_menu_color(self, colors: np.ndarray, titles: List[str], agg) -> Dict:
        colors_arr = np.array(colors)
        colors_num = colors_arr.shape[1] if colors_arr.ndim == 2 else 1

        def _colors_agg(i: int) -> dict:
            if i is None:
                arr = colors_arr
            else:
                arr = colors_arr[:, i] if colors_arr.ndim == 2 else colors_arr
            return aggregate_graph(arr, self.graph, agg)

        def _update_colors(i: int) -> dict:
            node_col_agg = _colors_agg(i)
            node_col_arr = list(node_col_agg.values())
            scatter_text = self._text(node_col_agg)
            cbar = self._colorbar(titles[i])
            if self.dim == 2:
                return {
                    "text": [scatter_text],
                    **{
                        f"marker.colorbar.{'.'.join(k.split('_'))}": [v]
                        for k, v in cbar.items()
                    },
                    "marker.color": [node_col_arr],
                    "marker.cmax": [max(node_col_arr, default=None)],
                    "marker.cmin": [min(node_col_arr, default=None)],
                }
            elif self.dim == 3:
                edge_col = self._edge_colors_from_node_colors(
                    node_col_agg,
                )
                return {
                    "text": [None, scatter_text],
                    **{
                        f"marker.colorbar.{'.'.join(k.split('_'))}": [None, v]
                        for k, v in cbar.items()
                    },
                    "marker.color": [None, node_col_arr],
                    "marker.cmax": [None, max(node_col_arr, default=None)],
                    "marker.cmin": [None, min(node_col_arr, default=None)],
                    "line.color": [edge_col, None],
                    "line.cmax": [max(node_col_arr, default=None), None],
                    "line.cmin": [min(node_col_arr, default=None), None],
                }
            return {}

        target_traces = [1] if self.dim == 2 else [0, 1]

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
            name=MENU_COLOR_NAME,
            buttons=buttons,
            direction="down",
            active=0,
            x=0.75,
            y=1.0,
            xanchor="center",
            yanchor="top",
        )

    def _ui_slider_node_size(self, node_sizes: List[float]) -> Dict:
        steps = [
            dict(
                method="restyle",
                label=f"{node_size}",
                args=[
                    {
                        "marker.sizeref": [
                            _MAX_SIZEREF if node_size == 0.0 else (1.0 / node_size) ** 2
                        ]
                    },
                    [1],
                ],
            )
            for node_size in node_sizes
        ]

        return dict(
            name=SLIDER_NODE_SIZE_NAME,
            active=len(steps) // 2,
            currentvalue=dict(
                prefix="Node size: ",
                visible=False,
                xanchor="center",
            ),
            steps=steps,
            x=0.5,
            y=0.0,
            xanchor="center",
            yanchor="bottom",
            len=0.5,
            lenmode="fraction",
            ticklen=1,
            pad=dict(t=1, b=1, l=1, r=1),
            bgcolor="rgba(1.0, 1.0, 1.0, 0.5)",
            activebgcolor="rgba(127, 127, 127, 0.5)",
        )
