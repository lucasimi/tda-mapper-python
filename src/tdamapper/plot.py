"""
This module provides functionalities to visualize the Mapper graph.
"""

from typing import Any, Callable, Literal, Optional, Union

import igraph as ig
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from tdamapper.plot_backends.plot_matplotlib import plot_matplotlib
from tdamapper.plot_backends.plot_plotly import plot_plotly, plot_plotly_update
from tdamapper.plot_backends.plot_pyvis import plot_pyvis

Dimensions = Literal[2, 3]
LayoutEngine = Literal["igraph", "networkx"]


class MapperPlot:
    """
    Class for generating and visualizing the Mapper graph.

    This class creates a metric embedding of the Mapper graph in 2D or 3D and
    converts it into a plot.

    :param graph: The precomputed Mapper graph to be embedded. This can be
        obtained by calling :func:`tdamapper.core.mapper_graph` or
        :func:`tdamapper.core.MapperAlgorithm.fit_transform`.
    :param dim: The dimension of the graph embedding (2 or 3).
    :param iterations: The number of iterations used to construct the graph
        embedding. Defaults to 50.
    :param seed: The random seed used to construct the graph embedding.
        Defaults to None.
    :param layout_engine: The engine used to compute the graph layout in the
        specified dimensions. Possible values are 'igraph' and 'networkx'.
        Defaults to 'igraph'.
    """

    def __init__(
        self,
        graph: nx.Graph,
        dim: Dimensions,
        iterations: int = 50,
        seed: Optional[int] = None,
        layout_engine: LayoutEngine = "igraph",
    ) -> None:
        self.graph = graph
        self.dim = dim
        self.iterations = iterations
        self.seed = seed
        self.layout_engine = layout_engine
        self.positions = self._compute_pos()

    def _compute_pos(self) -> dict[int, tuple[float, ...]]:
        if self.layout_engine == "igraph":
            return self._compute_pos_ig()
        if self.layout_engine == "networkx":
            return self._compute_pos_nx()
        raise ValueError(
            f"Unknown engine {self.layout_engine}. "
            "Only possible values are 'igraph' and 'networkx'"
        )

    def _compute_pos_nx(self) -> dict[int, tuple[float, ...]]:
        return nx.spring_layout(
            self.graph,
            dim=self.dim,
            seed=self.seed,
            iterations=self.iterations,
        )

    def _compute_pos_ig(self) -> dict[int, tuple[float, ...]]:
        if self.graph.number_of_nodes() == 0:
            return {}
        rng = np.random.default_rng(self.seed)
        random_pos = rng.random((len(self.graph.nodes), self.dim))
        graph_ig = ig.Graph.from_networkx(self.graph)
        pos: dict[int, tuple[float, ...]] = {}
        if self.dim == 2:
            layout = graph_ig.layout_fruchterman_reingold(
                niter=self.iterations,
                seed=random_pos,
            )
            pos = {
                node: (layout[i][0], layout[i][1])
                for i, node in enumerate(self.graph.nodes())
            }
        elif self.dim == 3:
            layout = graph_ig.layout_fruchterman_reingold_3d(
                niter=self.iterations,
                seed=random_pos,
            )
            pos = {
                node: (layout[i][0], layout[i][1], layout[i][2])
                for i, node in enumerate(self.graph.nodes())
            }
        return pos

    def plot_matplotlib(
        self,
        colors: NDArray[np.float_],
        node_size: int = 1,
        agg: Callable[..., Any] = np.nanmean,
        title: Optional[str] = None,
        cmap: str = "jet",
        width: int = 512,
        height: int = 512,
    ) -> tuple[Figure, Axes]:
        """
        Draw a static plot using Matplotlib.

        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :param node_size: A scaling factor for node size. Defaults to 1.
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to `numpy.nanmean`.
        :param title: The title to be displayed alongside the figure.
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to 'jet'.
        :param width: The desired width of the figure in pixels. Defaults to
            512.
        :param height: The desired height of the figure in pixels. Defaults to
            512

        :return: A static matplotlib figure that can be displayed on screen
            and notebooks.
        """

        return plot_matplotlib(
            self,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )

    def plot_plotly(
        self,
        colors: NDArray[np.float_],
        node_size: Union[int, float, list[Union[int, float]]] = 1,
        agg: Callable[..., Any] = np.nanmean,
        title: Optional[Union[str, list[str]]] = None,
        cmap: Union[str, list[str]] = "jet",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> go.Figure:
        """
        Draw an interactive plot using Plotly.

        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :param node_size: A scaling factor for node size. When node_size is a
            list, the figure will display a slider with the specified values.
            Defaults to 1.
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to `numpy.nanmean`.
        :param title: The title for the colormap. When colors has shape (n, m)
            and title is a list of string, each item will be used as title for
            its corresponding colormap.
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to 'jet'.
        :param width: The desired width of the figure in pixels.
        :param height: The desired height of the figure in pixels.

        :return: An interactive Plotly figure that can be displayed on screen
            and notebooks. For 3D embeddings, the figure requires a WebGL
            context to be shown.
        """
        return plot_plotly(
            self,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )

    def plot_plotly_update(
        self,
        fig: go.Figure,
        colors: Optional[NDArray[np.float_]] = None,
        node_size: Optional[Union[int, float, list[Union[int, float]]]] = None,
        agg: Optional[Callable[..., Any]] = None,
        title: Optional[str] = None,
        cmap: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> go.Figure:
        """
        Draw an interactive plot using Plotly on a previously rendered figure.

        This is typically faster than calling `MapperPlot.plot_plotly` on a
        new set of parameters.

        :param fig: A Plotly Figure object obtained by calling the method
            `MapperPlot.plot_plotly`.
        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data. Defaults to None.
        :param node_size: A scaling factor for node size. Defaults to None.
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to None.
        :param title: The title to be displayed alongside the figure. Defaults
            to None.
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to None.
        :param width: The desired width of the figure in pixels. Defaults to
            None.
        :param height: The desired height of the figure in pixels. Defaults to
            None.

        :return: An interactive Plotly figure that can be displayed on screen
            and notebooks. For 3D embeddings, the figure requires a WebGL
            context to be shown.
        """
        return plot_plotly_update(
            self,
            fig,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )

    def plot_pyvis(
        self,
        output_file: str,
        colors: NDArray[np.float_],
        node_size: int = 1,
        agg: Callable[..., Any] = np.nanmean,
        title: Optional[str] = None,
        cmap: str = "jet",
        width: int = 512,
        height: int = 512,
    ) -> None:
        """
        Draw an interactive HTML plot using PyVis.

        :param output_file: The path where the html file is written.
        :param colors: An array of values that determine the color of each
            node in the graph, useful for highlighting different features of
            the data.
        :param node_size: A scaling factor for node size. Defaults to 1.
        :param agg: A function used to aggregate the `colors` array over the
            points within a single node. The final color of each node is
            obtained by mapping the aggregated value with the colormap `cmap`.
            Defaults to `numpy.nanmean`.
        :param title: The title to be displayed alongside the figure. Defaults
            to None.
        :param cmap: The name of a colormap used to map `colors` data values,
            aggregated by `agg`, to actual RGBA colors. Defaults to 'jet'.
        :param width: The desired width of the figure in pixels. Defaults to
            512.
        :param height: The desired height of the figure in pixels. Defaults to
            512.
        """
        return plot_pyvis(
            self,
            output_file=output_file,
            colors=colors,
            node_size=node_size,
            agg=agg,
            title=title,
            cmap=cmap,
            width=width,
            height=height,
        )
