import panel as pn

from jinja2 import Template
import json
import time
import os

import pandas as pd
import numpy as np
import plotly.graph_objs as go 

import networkx as nx
from networkx.readwrite.json_graph.adjacency import adjacency_data
from networkx.readwrite.json_graph.node_link import node_link_data

from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm, aggregate_graph
from tdamapper.cover import CubicalCover, BallCover, TrivialCover
from tdamapper.clustering import TrivialClustering, FailSafeClustering
from tdamapper.plot import MapperLayoutInteractive


MAX_SAMPLES = 1000

SAMPLE_FRAC = 0.1


V_DATA_SUMMARY_FEAT = 'feature'

V_DATA_SUMMARY_MEAN = 'mean'

V_DATA_SUMMARY_VAR = 'var'

V_DATA_SUMMARY_VMR = 'vmr'

V_DATA_SUMMARY_HIST = 'histogram'

V_DATA_SUMMARY_COLOR = 'color'

V_DATA_SUMMARY_BINS = 5


def get_sample(df: pd.DataFrame, frac=SAMPLE_FRAC, max_n=MAX_SAMPLES, rand=42):
    if frac * len(df) > max_n:
        return df.sample(n=max_n, random_state=rand)
    return df.sample(frac=frac, random_state=rand)


def fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include='number')
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def get_data_summary(df_X, df_y):
    df = pd.concat([get_sample(df_y), get_sample(df_X)], axis=1)
    df_hist = pd.DataFrame({x: df[x].value_counts(bins=V_DATA_SUMMARY_BINS, sort=False).values for x in df.columns}).T
    df_summary = pd.DataFrame({
        V_DATA_SUMMARY_FEAT: df.columns,
        V_DATA_SUMMARY_HIST: df_hist.values.tolist()
    })
    df_summary[V_DATA_SUMMARY_COLOR] = False
    return df_summary


def get_data_summary_simple(df_X, df_y):
    df = pd.concat([get_sample(df_y), get_sample(df_X)], axis=1)
    df_mean = df.mean(skipna=True)
    df_var = df.var(skipna=True)
    df_vmr = df_var / df_mean
    df_summary = pd.DataFrame({
        V_DATA_SUMMARY_FEAT: df.columns,
        V_DATA_SUMMARY_MEAN: df_mean,
        V_DATA_SUMMARY_VAR: df_var,
        V_DATA_SUMMARY_VMR: df_vmr})
    df_summary[V_DATA_SUMMARY_COLOR] = False
    return df_summary


class Settings:

    def __init__(self):
        self.data_source_type = 'Example'
        self.data_source_example = 'digits'
        self.data_source_openml = ''
        self.data_source_upload = None
        self.lens_type = 'Identity'
        self.lens_pca_n = 1
        self.cover_type = 'Cubical'
        self.cover_cubical_n = 10
        self.cover_cubical_p = 0.5
        self.cover_ball_r = 100.0
        self.cover_ball_metric_p = 2
        self.clustering_type = 'Trivial'
        self.clustering_agglomerative_n = 2


class Results:

    def __init__(self):
        self.df_X = None
        self.df_y = None
        self.X = None
        self.df_summary = None
        self.lens = None
        self.mapper_graph = None
        self.mapper_plot = None

    def set_df(self, X, y):
        self.df_X = fix_data(X)
        self.df_y = fix_data(y)
        self.X = self.df_X.to_numpy()
        self.df_summary = get_data_summary_simple(self.df_X, self.df_y)

    def set_lens(self, lens):
        self.lens = lens

    def set_mapper_graph(self, mapper_graph):
        self.mapper_graph = mapper_graph

    def set_mapper_plot(self, mapper_plot):
        self.mapper_plot = mapper_plot

    def clear_df(self):
        self.df_X = None
        self.df_y = None
        self.X = None
        self.df_summary = None
        self.lens = None

    def clear_mapper(self):
        self.mapper_graph = None
        self.mapper_plot = None

    def clear(self):
        self.clear_df()
        self.clear_mapper()


def data_source(settings):
    ds_type = pn.widgets.Select(
        options=['Example', 'OpenML', 'Upload'],
        value=settings.data_source_type)
    ds_example = pn.widgets.Select(
        options=['digits', 'iris'],
        value=settings.data_source_example)
    ds_openml = pn.widgets.TextInput(
        value=settings.data_source_openml)
    ds_upload = pn.widgets.FileInput(
        value=settings.data_source_upload)

    def _ds_select(opt):
        settings.data_source_type = opt
        if opt == 'Example':
            return ds_example
        elif opt == 'OpenML':
            return ds_openml
        elif opt == 'Upload':
            return ds_upload
        else:
            return pn.Column()

    def _ds_set_example(ds_example):
        settings.data_source_example = ds_example

    def _ds_set_openml(ds_openml):
        settings.data_source_openml = ds_openml

    def _ds_set_upload(ds_upload):
        settings.data_source_upload = ds_upload

    ds_select = pn.bind(_ds_select, ds_type)
    pn.bind(_ds_set_example, ds_example, watch=True)
    pn.bind(_ds_set_openml, ds_openml, watch=True)
    pn.bind(_ds_set_upload, ds_upload, watch=True)
    return pn.Column(
        ds_type,
        ds_select)


def lens_settings(settings):
    lens_type = pn.widgets.Select(
        name='lens',
        options=['Identity', 'PCA'],
        value=settings.lens_type)
    lens_pca_n = pn.widgets.NumberInput(
        name='n',
        value=settings.lens_pca_n)

    def _lens_select(t):
        settings.lens_type = t
        if t == 'PCA':
            return lens_pca_n
        else:
            return pn.Column()

    def _lens_set_pca(lens_pca_n):
        settings.lens_pca_n = lens_pca_n

    pn.bind(_lens_set_pca, lens_pca_n, watch=True)
    lens_select = pn.bind(_lens_select, lens_type)
    return pn.Column(
        lens_type,
        lens_select)


def cover_settings(settings):
    cover_type = pn.widgets.Select(
        options=['Cubical', 'Ball', 'Trivial'],
        value=settings.cover_type)
    cover_cubical_n = pn.widgets.NumberInput(
        name='n',
        value=settings.cover_cubical_n)
    cover_cubical_p = pn.widgets.NumberInput(
        name='overlap',
        value=settings.cover_cubical_p)
    cover_ball_r = pn.widgets.NumberInput(
        name='radius',
        value=settings.cover_ball_r)
    cover_ball_metric_p = pn.widgets.NumberInput(
        name='Lp',
        value=settings.cover_ball_metric_p)

    def _cover_select(opt):
        settings.cover_type = opt
        if opt == 'Cubical':
            return pn.Column(
                cover_cubical_n,
                cover_cubical_p)
        elif opt == 'Ball':
            return pn.Column(
                cover_ball_r,
                cover_ball_metric_p)
        else:
            return pn.Column()

    def _set_cover_cubical_n(cover_cubical_n):
        print('cubical_n', cover_cubical_n)
        settings.cover_cubical_n = cover_cubical_n

    def _set_cover_cubical_p(cover_cubical_p):
        print('cubical_p', cover_cubical_p)
        settings.cover_cubical_p = cover_cubical_p

    def _set_cover_ball_r(cover_ball_r):
        print('ball_r', cover_ball_r)
        settings.cover_ball_r = cover_ball_r

    def _set_cover_ball_metric_p(cover_ball_metric_p):
        print('ball_metric_p', cover_ball_metric_p)
        settings.cover_ball_metric_p = cover_ball_metric_p

    cover_select = pn.bind(_cover_select, cover_type)
    pn.bind(_set_cover_cubical_n, cover_cubical_n, watch=True)
    pn.bind(_set_cover_cubical_p, cover_cubical_p, watch=True)
    pn.bind(_set_cover_ball_r, cover_ball_r, watch=True)
    pn.bind(_set_cover_ball_metric_p, cover_ball_metric_p, watch=True)
    return pn.Column(
        cover_type,
        cover_select)


def clustering_settings(settings):
    clustering_type = pn.widgets.Select(
        options=['Trivial', 'Agglomerative'],
        value=settings.clustering_type)
    clustering_agglomerative_n = pn.widgets.NumberInput(
        name='n',
        value=settings.clustering_agglomerative_n)

    def _clustering_select(opt):
        if opt == 'Agglomerative':
            return clustering_agglomerative_n
        else:
            return pn.Column()

    def _set_clustering_agglomerative_n(clustering_agglomerative_n):
        settings.clustering_agglomerative_n = clustering_agglomerative_n

    clustering_select = pn.bind(_clustering_select, clustering_type)
    pn.bind(_set_clustering_agglomerative_n, clustering_agglomerative_n, watch=True)

    return pn.Column(
        clustering_type,
        clustering_select)


def sample_3d_plot():
    t = np.linspace(0, 10, 50)
    x, y, z = np.cos(t), np.sin(t), t
    fig = go.Figure(
        data=go.Scatter3d(x=x, y=y, z=z, mode='markers'),
        layout=dict(title='3D Scatter Plot'))
    return fig


def main():
    results = Results()
    settings = Settings()
    pn.extension('tabulator')
    pn.extension(sizing_mode='stretch_width')
    ds_load = pn.widgets.Button(
        name='Load')
    ds_df = pn.widgets.Tabulator(
        pd.DataFrame(),
        show_index=False)
    run = pn.widgets.Button(name='Run')
    graph_pane = pn.pane.HTML('<iframe width="100%" height="100%"></iframe>')

    def _load_ds(ds_load):
        if ds_load:
            if settings.data_source_type == 'Example':
                if settings.data_source_example == 'digits':
                    print('loading digits')
                    X, y = load_digits(return_X_y=True, as_frame=True)
                    results.set_df(X, y)
                    ds_df.value = results.df_summary
                elif settings.data_source_example == 'iris':
                    print('loading iris')
                    X, y = load_iris(return_X_y=True, as_frame=True)
                    results.set_df(X, y)
                    ds_df.value = results.df_summary

    def _compute_mapper(run, mapper_pane):
        if not run:
            return
        print('computing')
        cover = TrivialCover()
        lens = results.X
        if settings.lens_type == 'PCA':
            print('settings.pca_n', settings.lens_pca_n)
            lens = PCA(settings.lens_pca_n).fit_transform(results.X)
        if settings.cover_type == 'Cubical':
            print('settings.cubical_n', settings.cover_cubical_n)
            print('settings.cubical_p', settings.cover_cubical_p)
            cover = CubicalCover(
                n_intervals=settings.cover_cubical_n,
                overlap_frac=settings.cover_cubical_p)
        elif settings.cover_type == 'Ball':
            print('settings.ball_r', settings.cover_ball_r)
            print('settings.ball_metric_p', settings.cover_ball_metric_p)
            cover = BallCover(
                radius=settings.cover_ball_r,
                metric=lambda x,y: np.linalg.norm(x - y, ord=settings.cover_ball_metric_p))
        clustering = TrivialClustering()
        if settings.clustering_type == 'Agglomerative':
            print('settings.agglomerative_n', settings.clustering_agglomerative_n)
            clustering = AgglomerativeClustering(
                n_clusters=settings.clustering_agglomerative_n)
        mapper_algo = MapperAlgorithm(
            cover=cover,
            clustering=FailSafeClustering(
                clustering=clustering))
        mapper_graph = mapper_algo.fit_transform(results.X, lens)
        results.set_mapper_graph(mapper_graph)
        source = 'assets/graph.html'
        with open(source, 'w') as outfile:
            outfile.write(graph_html(mapper_graph, results.df_y.to_numpy()))
        mapper_pane.object = f'<iframe src={source}?time={int(time.time())} width="100%" height="100%"></iframe>'
        print('computed')

    gspec = pn.GridSpec(sizing_mode='stretch_both')

    gspec[:, 0] = pn.Column(
        pn.Tabs(
            ('Data Source', pn.Column(
                data_source(settings),
                ds_load,
                ds_df)),
            ('Mapper Settings', pn.Column(
                pn.pane.Markdown('### Lens'),
                lens_settings(settings),
                pn.pane.Markdown('### Cover'),
                cover_settings(settings),
                pn.pane.Markdown('### Clustering'),
                clustering_settings(settings),
                run))),
        #pn.pane.Markdown('### Data Source'),
        #data_source(settings),
        width=300)

    gspec[:, 1:4] = graph_pane

    pn.bind(_load_ds, ds_load, watch=True)
    pn.bind(_compute_mapper, run, graph_pane, watch=True)
    app = pn.Column(
        pn.Row(pn.pane.Markdown('# tda-mapper app')),
        gspec)
    app.servable()


def graph_html(graph, colors):
    g = graph.copy()
    colors_agg = aggregate_graph(colors, g, agg=np.nanmean)
    nx.set_node_attributes(g, colors_agg, name='color')
    g_json = node_link_data(g)
    template = """
    <!DOCTYPE html>
    <html lang="en-US">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Today's Date</title>
            <script src="http://d3js.org/d3.v7.min.js"></script>
            <style>
                .node {stroke: #fff; stroke-width: 1.5px;}
                .link {stroke: #999; stroke-opacity: .6;}
                body, html {
                    padding:0;
                    margin:0;
                    height: 100%;
                    width: 100%;
                }
            </style>
        </head>
        <body>
            <script>
                const graph_json = {{ graph|tojson|safe }};
            </script>
            <svg width="100%" height="100%"></svg>
            <script>
                {{ js_code }}
            </script>
        </body>
    </html>
    """
    with open('assets/graph.js', 'r', encoding='utf-8') as js_source:
        js_code = js_source.read()
    j2_template = Template(template)
    return j2_template.render(
        graph=g_json,
        js_code=js_code)



main()
