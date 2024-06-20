import json
import time
import io
import gzip

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import networkx as nx 
from networkx.readwrite.json_graph import adjacency_data

from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm, ATTR_SIZE
from tdamapper.cover import CubicalCover, BallCover, TrivialCover
from tdamapper.clustering import TrivialClustering, FailSafeClustering
from tdamapper.plot import MapperLayoutInteractive


MAX_NODES = 500

MAX_SAMPLES = 1000

SAMPLE_FRAC = 0.1

OPENML_URL = 'https://www.openml.org/search?type=data&sort=runs&status=active'

DATA_INFO = 'Non-numeric and NaN features get dropped. NaN rows get replaced by mean'

GIT_REPO_URL = 'https://github.com/lucasimi/tda-mapper-python'

REPORT_BUG = f'{GIT_REPO_URL}/issues'

ABOUT = f'{GIT_REPO_URL}/blob/main/README.md'

DESCRIPTION = '''
    Gain insights from your data with the *Mapper Algorithm*
    '''

FOOTER = f'''
    If you find this app useful, please consider leaving a :star: on **[GitHub]({GIT_REPO_URL})**.
    '''

ICON_URL = f'{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-icon.png'

LOGO_URL = f'{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png'

APP_TITLE = 'TDA Mapper App'

# V_* are reusable values for widgets

V_LENS_IDENTITY = 'Identity'

V_LENS_PCA = 'PCA'

V_COVER_TRIVIAL = 'Trivial'

V_COVER_BALL = 'Ball'

V_COVER_CUBICAL = 'Cubical'

V_CLUSTERING_TRIVIAL = 'Trivial'

V_CLUSTERING_AGGLOMERATIVE = 'Agglomerative'

V_DATA_SUMMARY_FEAT = 'feature'

V_DATA_SUMMARY_HIST = 'histogram'

V_DATA_SUMMARY_COLOR = 'color'

V_DATA_SUMMARY_BINS = 5

# VD_* are reusable default values for widgets

VD_SEED = 42

VD_DIM = 3

# S_* are reusable manually managed stored objects

S_RESULTS = 'stored_results'


class Results:

    def __init__(self):
        self.df_X = pd.DataFrame()
        self.df_y = pd.DataFrame()
        self.df_X_sample = pd.DataFrame()
        self.df_y_sample = pd.DataFrame()
        self.df_all = pd.DataFrame()
        self.X = self.df_X.to_numpy()
        self.df_summary = pd.DataFrame()
        self.mapper_graph = nx.Graph()
        self.mapper_plot = None
        self.mapper_fig = self._init_fig()
        self.auto_rendering = None

    def _init_fig(self):
        fig = go.Figure(
            data=[go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode='markers')])
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showgrid=True,
                    zeroline=True,
                    showline=True,
                    ticks='outside'),
                yaxis=dict(
                    showgrid=True,
                    zeroline=True,
                    showline=True,
                    ticks='outside'),
                zaxis=dict(
                    showgrid=True,
                    zeroline=True,
                    showline=True,
                    ticks='outside')))
        return fig

    def set_df(self, X, y):
        self.df_X = fix_data(X)
        self.df_y = fix_data(y)
        self.df_X_sample = get_sample(self.df_X)
        self.df_y_sample = get_sample(self.df_y)
        self.df_all = pd.concat([self.df_y, self.df_X], axis=1)
        self.X = self.df_X.to_numpy()
        self.df_summary = _get_data_summary(self.df_X, self.df_y)
        self.mapper_graph = nx.Graph()
        self.mapper_plot = None
        self.mapper_fig = self._init_fig()
        self.auto_rendering = None

    def set_mapper(self, mapper_graph):
        self.mapper_graph = mapper_graph
        self.mapper_plot = MapperLayoutInteractive(
            self.mapper_graph,
            dim=VD_DIM,
            height=450,
            width=450,
            colors=self.X,
            seed=VD_SEED)
        self.mapper_fig = go.Figure()
        nodes_num = mapper_graph.number_of_nodes()
        if nodes_num <= MAX_NODES:
            self.auto_rendering = True
        else:
            self.auto_rendering = False

    def set_mapper_fig(self, mapper_fig):
        self.mapper_fig = mapper_fig
        self.auto_rendering = None


def lp_metric(p):
    return lambda x, y: np.linalg.norm(x - y, ord=p)


def fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include='number')
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def get_gzip_bytes(string, encoding='utf-8'):
    fileobj = io.BytesIO()
    gzf = gzip.GzipFile(fileobj=fileobj, mode='wb', compresslevel=6)
    gzf.write(string.encode(encoding))
    gzf.close()
    return fileobj.getvalue()


@st.cache_data
def get_sample(df: pd.DataFrame, frac=SAMPLE_FRAC, max_n=MAX_SAMPLES, rand=42):
    if frac * len(df) > max_n:
        return df.sample(n=max_n, random_state=rand)
    return df.sample(frac=frac, random_state=rand)


@st.cache_data
def cached_load_digits():
    return load_digits(return_X_y=True, as_frame=True)


@st.cache_data
def cached_load_iris():
    return load_iris(return_X_y=True, as_frame=True)


@st.cache_data
def cached_fetch_openml(source):
    return fetch_openml(source, return_X_y=True, as_frame=True)


@st.cache_data
def df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')


def _get_data_summary(df_X, df_y):
    df = pd.concat([get_sample(df_y), get_sample(df_X)], axis=1)
    df_hist = pd.DataFrame({x: df[x].value_counts(bins=V_DATA_SUMMARY_BINS, sort=False).values for x in df.columns}).T
    df_summary = pd.DataFrame({
        V_DATA_SUMMARY_FEAT: df.columns,
        V_DATA_SUMMARY_HIST: df_hist.values.tolist()})
    df_summary[V_DATA_SUMMARY_COLOR] = False
    return df_summary


def _mapper_caption():
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    nodes_num = 0
    edges_num = 0
    if mapper_graph is not None:
        nodes_num = mapper_graph.number_of_nodes()
        edges_num = mapper_graph.number_of_edges()
    cap = f'{nodes_num} nodes, {edges_num} edges'
    st.caption(cap)


def _mapper_histogram():
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    ccs = nx.connected_components(mapper_graph)
    size = nx.get_node_attributes(mapper_graph, ATTR_SIZE)
    node_cc, node_size = {}, {}
    node_cc_max, node_size_max = 1, 1
    for cc in ccs:
        cc_len = len(cc)
        for u in cc:
            u_size = size[u]
            node_cc[u] = cc_len
            node_size[u] = u_size
            if u_size > node_size_max:
                node_size_max = u_size
        if cc_len > node_cc_max:
            node_cc_max = cc_len
    arr_size = np.array([node_size[u]/node_size_max for u in mapper_graph.nodes()])
    arr_cc = np.array([node_cc[u]/node_cc_max for u in mapper_graph.nodes()])
    df = pd.DataFrame(dict(
        series=np.concatenate((
            ['node size (rel.)'] * len(arr_size),
            ['conn. comp. size (rel.)'] * len(arr_cc))),
        data=np.concatenate((
            arr_size,
            arr_cc))))
    fig = px.histogram(
        df,
        nbins=20,
        color='series',
        histnorm='probability density',
        opacity=0.75,
        barmode='overlay')
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        xaxis_visible=True,
        xaxis_title_standoff=10,
        xaxis_title=None,
        yaxis_title_standoff=10,
        yaxis_visible=True,
        yaxis_title=None,
        legend=dict(
            title_text=None,
            yanchor='top',
            orientation='v',
            y=0.99,
            xanchor='left',
            x=0.01,
            bordercolor='#d5d6d8',
            borderwidth=1))
    with st.container(border=False):
        st.plotly_chart(
            fig,
            use_container_width=True,
            config = {'displayModeBar': False})


def _mapper_download():
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    mapper_adj = {} if mapper_graph is None else adjacency_data(mapper_graph)
    mapper_json = json.dumps(mapper_adj, default=int)
    nodes_num = mapper_graph.number_of_nodes()
    return st.download_button(
        '📥 Download Mapper',
        data=get_gzip_bytes(mapper_json),
        disabled=nodes_num < 1,
        use_container_width=True,
        file_name=f'mapper_graph_{int(time.time())}.json.gzip')


def _initialize_state():
    if S_RESULTS not in st.session_state:
        st.session_state[S_RESULTS] = Results()


def _initialize_page():
    st.set_page_config(
        layout='wide',
        page_icon=ICON_URL,
        page_title=APP_TITLE,
        menu_items={
            'Report a bug': REPORT_BUG,
            'About': ABOUT})


def _initialize_sidebar():
    with st.sidebar:
        st.image(LOGO_URL)
        st.markdown(DESCRIPTION)
        st.link_button(
            'More on **GitHub**',
            url=GIT_REPO_URL,
            use_container_width=True,
            type='primary')
        st.subheader('#')


def initialize():
    _initialize_state()
    _initialize_page()
    _initialize_sidebar()


def _update_data(data_source):
    X, y = pd.DataFrame(), pd.DataFrame()
    if isinstance(data_source, io.BytesIO):
        X, y = pd.read_csv(data_source), pd.DataFrame()
    elif data_source == 'Digits':
        X, y = cached_load_digits()
    elif data_source == 'Iris':
        X, y = cached_load_iris()
    elif isinstance(data_source, str):
        try:
            X, y = cached_fetch_openml(data_source)
        except ValueError as err:
            st.toast(f'# {err}', icon='🚨')
    df_X, df_y = fix_data(X), fix_data(y)
    st.session_state[S_RESULTS].set_df(df_X, df_y)
    st.toast('Successfully Loaded Data', icon='✅')


def _data_caption():
    df_X = st.session_state[S_RESULTS].df_X
    df_y = st.session_state[S_RESULTS].df_y
    if df_X.empty:
        cap = 'Empty dataset'
    else:
        inst = len(df_X)
        feats = len(df_X.columns) + len(df_y.columns)
        cap = f'{inst} instances, {feats} features'
    st.caption(cap)


def _data_summary():
    df_all = st.session_state[S_RESULTS].df_all
    st.dataframe(
        df_all.head(50),
        use_container_width=True,
        height=250)


def _data_download():
    df_all = st.session_state[S_RESULTS].df_all
    df_all_data = df_to_csv(df_all)
    st.download_button(
        '📥 Download Data',
        disabled=df_all.empty,
        use_container_width=True,
        data=df_all_data,
        file_name='dataset.csv',
        mime='text/csv')


def data_output_section():
    _data_caption()
    _data_summary()
    _data_download()


def data_input_section():
    data_source_type = st.selectbox(
        'Source',
        options=['Example', 'OpenML', 'CSV'])
    if data_source_type == 'Example':
        data_source = st.selectbox('Name', options=['Digits', 'Iris'])
    elif data_source_type == 'OpenML':
        data_source = st.text_input('Name', placeholder='Name', help=f'Search on [OpenML]({OPENML_URL})')
    elif data_source_type == 'CSV':
        data_source = st.file_uploader('Upload')
    load_button = st.button(
        '📦 Load',
        use_container_width=True)
    if load_button:
        _update_data(data_source)


def _update_mapper(X, lens, cover, clustering):
    if (X is None) or (lens is None) or (cover is None) or (clustering is None):
        st.warning('Make sure you selected the right options')
    mapper_algo = MapperAlgorithm(
        cover=cover,
        clustering=FailSafeClustering(
            clustering=clustering,
            verbose=False))
    mapper_graph = mapper_algo.fit_transform(X, lens)
    st.session_state[S_RESULTS].set_mapper(mapper_graph)
    st.toast('Successfully Computed Mapper', icon='✅')
    auto_rendering = st.session_state[S_RESULTS].auto_rendering
    if auto_rendering is False:
        st.toast('Automatic Rendering Disabled: Graph Too Large', icon='⚠️')


def mapper_settings_section():
    lens_type = st.selectbox(
        '🔎 Lens',
        options=[V_LENS_IDENTITY, V_LENS_PCA],
        index=1)
    cover_type = st.selectbox(
        '🌐 Cover',
        options=[V_COVER_TRIVIAL, V_COVER_BALL, V_COVER_CUBICAL],
        index=2)
    clustering_type = st.selectbox(
        '🧮 Clustering',
        options=[V_CLUSTERING_TRIVIAL, V_CLUSTERING_AGGLOMERATIVE],
        index=1)
    return lens_type, cover_type, clustering_type


def _lens_tuning(lens_type):
    X = st.session_state[S_RESULTS].X
    lens = None
    if lens_type == V_LENS_IDENTITY:
        lens = X
    elif lens_type == V_LENS_PCA:
        pca_n = st.number_input(
            '🔎 PCA Components',
            value=2,
            min_value=1)
        _, n_feats = X.shape
        if pca_n > n_feats:
            lens = X
        else:
            lens = PCA(n_components=pca_n).fit_transform(X)
    return lens


def _cover_tuning(cover_type):
    cover = None
    if cover_type == V_COVER_TRIVIAL:
        cover = TrivialCover()
    elif cover_type == V_COVER_BALL:
        ball_r = st.number_input(
            '🌐 Radius',
            value=100.0,
            min_value=0.0)
        ball_metric_p = st.number_input(
            '🌐 Lp Metric',
            value=2,
            min_value=1)
        cover = BallCover(radius=ball_r, metric=lp_metric(ball_metric_p))
    elif cover_type == V_COVER_CUBICAL:
        cubical_n = st.number_input(
            '🌐 Intervals',
            value=10,
            min_value=0)
        cubical_p = st.number_input(
            '🌐 Overlap Fraction',
            value=0.25,
            min_value=0.0,
            max_value=1.0)
        cover = CubicalCover(n_intervals=cubical_n, overlap_frac=cubical_p)
    return cover


def _clustering_tuning(clustering_type):
    clustering = None
    if clustering_type == V_CLUSTERING_TRIVIAL:
        clustering = TrivialClustering()
    elif clustering_type == V_CLUSTERING_AGGLOMERATIVE:
        clust_n = st.number_input(
            '🧮 Clusters',
            value=2,
            min_value=1)
        clustering = AgglomerativeClustering(n_clusters=clust_n)
    return clustering


def mapper_run_section(lens_type, cover_type, clustering_type):
    X = st.session_state[S_RESULTS].X
    lens = _lens_tuning(lens_type)
    cover = _cover_tuning(cover_type)
    clustering = _clustering_tuning(clustering_type)
    run_button = st.button(
        '🚀 Run',
        use_container_width=True,
        disabled=X.size == 0)
    if run_button:
        _update_mapper(X, lens, cover, clustering)


def mapper_output_section():
    _mapper_caption()
    _mapper_histogram()
    _mapper_download()


def _update_fig(seed, colors, agg):
    mapper_plot = st.session_state[S_RESULTS].mapper_plot
    if mapper_plot is None:
        return
    mapper_plot.update(
        colors=colors,
        seed=seed,
        agg=agg)
    mapper_fig = mapper_plot.plot()
    mapper_fig.update_layout(
        uirevision='constant',
        margin=dict(b=0, l=0, r=0, t=0))
    st.session_state[S_RESULTS].set_mapper_fig(mapper_fig)
    st.toast('Successfully Rendered Graph', icon='✅')


def _update_auto_rendering():
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    nodes_num = mapper_graph.number_of_nodes() if mapper_graph else 0
    if nodes_num <= MAX_NODES:
        st.session_state[S_RESULTS].auto_rendering = True


def _mapper_colors():
    X = st.session_state[S_RESULTS].X
    df_X = st.session_state[S_RESULTS].df_X
    df_y = st.session_state[S_RESULTS].df_y
    df_all = st.session_state[S_RESULTS].df_all
    df_summary = st.session_state[S_RESULTS].df_summary
    colors = X
    data_edit = st.data_editor(
        df_summary,
        height=250,
        hide_index=True,
        disabled=(c for c in df_summary.columns if c != V_DATA_SUMMARY_COLOR),
        use_container_width=True,
        column_config={
            V_DATA_SUMMARY_HIST: st.column_config.AreaChartColumn(
                width='small'),
            V_DATA_SUMMARY_FEAT: st.column_config.TextColumn(
                width='small',
                disabled=True),
            V_DATA_SUMMARY_COLOR: st.column_config.CheckboxColumn(
                width='small',
                disabled=False)
        }, on_change=_update_auto_rendering)
    if not data_edit.empty:
        color_features = data_edit[data_edit[V_DATA_SUMMARY_COLOR]][V_DATA_SUMMARY_FEAT]
        if not color_features.empty:
            selected = pd.concat([df_all[c] for c in color_features], axis=1)
            if not selected.empty:
                colors = selected.to_numpy()
    return colors


def _mapper_agg():
    agg_sel = st.selectbox(
        'Aggregation',
        options=['Mean', 'Std'],
        on_change=_update_auto_rendering)
    if agg_sel == 'Mean':
        agg = np.mean
    elif agg_sel == 'Std':
        agg = np.std
    return agg


def _mapper_seed():
    seed = st.number_input(
        'Seed',
        value=VD_SEED,
        help='Changing this value alters the shape',
        on_change=_update_auto_rendering)
    return seed


def mapper_draw_section():
    colors = _mapper_colors()
    agg = _mapper_agg()
    seed = _mapper_seed()
    auto_rendering = st.session_state[S_RESULTS].auto_rendering
    if auto_rendering:
        _update_fig(seed, colors, agg)
    else:
        mapper_plot = st.session_state[S_RESULTS].mapper_plot
        update_button = st.button(
            '🌊 Update',
            use_container_width=True,
            disabled=mapper_plot is None)
        if update_button:
            _update_fig(seed, colors, agg)


def mapper_rendering_section():
    mapper_fig = st.session_state[S_RESULTS].mapper_fig
    with st.container(border=False):
        st.plotly_chart(
            mapper_fig,
            height=350,
            use_container_width=True)


def main():
    initialize()
    with st.sidebar:
        data_input_section()
        with st.popover('More', use_container_width=True):
            data_output_section()
    col_0, col_1 = st.columns([1, 3])
    with col_0:
        lens_type, cover_type, clustering_type = mapper_settings_section()
        with st.popover('🚀 Run', use_container_width=True):
            mapper_run_section(lens_type, cover_type, clustering_type)
        with st.popover('🎨 Draw', use_container_width=True):
            mapper_draw_section()
        with st.popover('More', use_container_width=True):
            mapper_output_section()
    with col_1:
        mapper_rendering_section()
    st.divider()
    st.markdown(FOOTER)


main()
