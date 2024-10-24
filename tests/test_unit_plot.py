import unittest

import numpy as np

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import BallCover
from tdamapper.clustering import TrivialClustering
from tdamapper.plot import MapperPlot


class TestMapperPlot(unittest.TestCase):

    def test_two_connected_clusters(self):
        data = [
            np.array([0.0, 1.0]), np.array([1.0, 0.0]),
            np.array([0.0, 0.0]), np.array([1.0, 1.0])
        ]
        mp = MapperAlgorithm(
            cover=BallCover(1.1, metric='euclidean'),
            clustering=TrivialClustering()
        )
        g = mp.fit_transform(data, data)
        mp_plot1 = MapperPlot(
            g,
            dim=2,
            seed=123,
            iterations=10
        )
        fig1 = mp_plot1.plot_plotly(
            colors=data,
            agg=np.nanmax,
            width=200,
            height=200,
            title='example',
            cmap='jet'
        )
        mp_plot2 = MapperPlot(
            g,
            dim=3,
            seed=123,
            iterations=10
        )
        fig2 = mp_plot2.plot_plotly(
            colors=data,
            agg=np.nanmax,
            width=200,
            height=200,
            title='example',
            cmap='jet'
        )
        fig3 = mp_plot2.plot_plotly_update(
            fig2,
            colors=data,
            agg=np.nanmin,
            width=300,
            height=300,
            title='example-updated',
            cmap='viridis'
        )
        mp_plot3 = MapperPlot(
            g,
            dim=2
        )
        fig4 = mp_plot3.plot_matplotlib(
            width=300,
            height=300,
            colors=data
        )
