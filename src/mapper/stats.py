import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

import mapper.pipeline

DPIS = 96

class MapperStats:

    def __init__(self, graph, kpis):
        self.__graph = graph
        self.__kpis = {} if not kpis else kpis

    def compute(self, data):
        for kpi_name, kpi_fun in self.__kpis.items():
            kpi_val = mapper.pipeline.aggregate(self.__graph, data, kpi_fun)
            nx.set_node_attributes(self.__graph, kpi_val, kpi_name)

    def plot_hist(self, width, height):
        df = pd.DataFrame({kpi_name: list(nx.get_node_attributes(self.__graph, kpi_name).values()) for kpi_name in self.__kpis})
        fig, axs = plt.subplots(1, len(self.__kpis), sharey=True, tight_layout=True, figsize=(width / DPIS, height / DPIS), dpi=DPIS)
        i = 0
        for kpi_name in self.__kpis:
            axs[i].hist(df[kpi_name], bins=10)
            axs[i].title.set_text(kpi_name)
            i += 1
        return fig
