import math

import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from mapper.core import aggregate_graph

def _rmse(x, y):
    return math.sqrt(mse(x, y))

MAPE = mape
MAE = mae
RMSE = _rmse


class MapperKpis:

    def __init__(self, X, graph): 
        self.__X = X
        self.__graph = graph 
        self.__kpis = {x: 0.5 for x in self.__graph.nodes()}

    def aggregate(self, metric, fun=lambda x: x, agg=np.nanmean):
        kpis = aggregate_graph(self.__X, self.__graph, metric, fun, agg)
        self.__kpis = kpis

    def plot(self, width, height, title=''):
        dpis = mapper.plot._DPIS
        fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(width / dpis, height / dpis), dpi=dpis)
        fig.patch.set_alpha(0.0)
        fig.subplots_adjust(bottom=0.0, right=1.0, top=1.0, left=0.0)
        ax.hist(self.__kpis.values(), bins=10, edgecolor='#ff4b4b', fc=(0, 0, 0, 0.0), linewidth=2)
        ax.title.set_text(title)
        ax.title.set_color(mapper.plot.EDGE_COLOR)
        ax.patch.set_alpha(0.0)
        for axis in ['top', 'right']:
            ax.spines[axis].set_linewidth(0)
        for axis in ['left', 'bottom']:
            ax.spines[axis].set_color(mapper.plot.EDGE_COLOR)
        ax.tick_params(axis='x', colors=mapper.plot.EDGE_COLOR)
        ax.tick_params(axis='y', colors=mapper.plot.EDGE_COLOR)
        ax.yaxis.label.set_color(mapper.plot.EDGE_COLOR)
        ax.xaxis.label.set_color(mapper.plot.EDGE_COLOR)
        return fig
