import unittest

import networkx as nx
import numpy as np

from tdamapper.core import TrivialClustering
from tdamapper.cover import BallCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot


class TestMapperPlot(unittest.TestCase):

    def test_two_connected_clusters(self):
        data = [
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
        ]
        mp = MapperAlgorithm(
            cover=BallCover(1.1, metric="euclidean"), clustering=TrivialClustering()
        )
        g = mp.fit_transform(data, data)
        mp_plot1 = MapperPlot(g, dim=2, seed=123, iterations=10)
        mp_plot1.plot_plotly(
            colors=data,
            agg=np.nanmax,
            width=200,
            height=200,
            title="example",
            cmap="jet",
        )
        mp_plot2 = MapperPlot(g, dim=3, seed=123, iterations=10)
        fig2 = mp_plot2.plot_plotly(
            colors=data,
            agg=np.nanmax,
            width=200,
            height=200,
            title="example",
            cmap="jet",
        )
        mp_plot2.plot_plotly_update(
            fig2,
            colors=data,
            agg=np.nanmin,
            width=300,
            height=300,
            title="example-updated",
            cmap="viridis",
        )
        mp_plot3 = MapperPlot(g, dim=2)
        mp_plot3.plot_matplotlib(width=300, height=300, colors=data)
        mp_plot3.plot_pyvis(
            width=512,
            height=512,
            colors=data,
            output_file="network.html",
        )

    def test_empty_graph(self):
        empty_graph = nx.Graph()
        mapper_plot = MapperPlot(empty_graph, dim=2)
        mapper_plot.plot_matplotlib(colors=[])
        mapper_plot.plot_plotly(colors=[])
        mapper_plot.plot_pyvis(colors=[], output_file="tmp.html")

    def test_two_connected_clusters_deprecated(self):
        data = [
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
        ]
        mapper_algo = MapperAlgorithm(
            cover=BallCover(1.1, metric="euclidean"), clustering=TrivialClustering()
        )
        g = mapper_algo.fit_transform(data, data)
        mapper_plot_1 = MapperPlot(
            g,
            dim=2,
            seed=123,
            iterations=10,
        )
        mapper_plot_1.plot_plotly(
            colors=data,
            agg=np.nanmax,
            width=200,
            height=200,
            title="example",
            cmap="jet",
        )
        mapper_plot_2 = MapperPlot(
            g,
            dim=3,
            seed=123,
            iterations=10,
        )
        fig = mapper_plot_2.plot_plotly(
            colors=data,
            agg=np.nanmax,
            width=200,
            height=200,
            title="example",
            cmap="jet",
        )
        mapper_plot_2.plot_plotly_update(
            fig,
            colors=data,
            agg=np.nanmin,
            width=300,
            height=300,
            title="example-updated",
            cmap="viridis",
        )
        mapper_plot_3 = MapperPlot(g, dim=2)
        mapper_plot_3.plot_matplotlib(
            colors=data,
        )
