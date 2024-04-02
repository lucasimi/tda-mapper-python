import json
import time
import io
import gzip
import random

import streamlit as st
import pandas as pd
import numpy as np


from networkx.readwrite.json_graph import adjacency_data

from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover, BallCover, TrivialCover
from tdamapper.clustering import TrivialClustering, FailSafeClustering
from tdamapper.plot import MapperLayoutInteractive


MAX_NODES = 1000

MAX_SAMPLES = 1000

SAMPLE_FRAC = 0.1

OPENML_URL = 'https://www.openml.org/search?type=data&sort=runs&status=active'

DATA_HELP = f'''
    To begin select you data source: 
    
    * Select an example to see how this works 
    * You can submit a csv to try on you data
    * Or you can use a publicly available dataset from [OpenML]({OPENML_URL}).
'''

DATA_INFO = 'Non-numeric and NaN features get dropped. NaN rows get replaced by mean'

MAPPER_HELP = "Experiment with Mapper Settings and hit Run when you're ready!"

MAPPER_PROCEED = '💣 Go on and show me the damn graph!'

MAPPER_COMPUTED = 'Mapper Graph Computed!'

GIT_REPO_URL = 'https://github.com/lucasimi/tda-mapper-python'

REPORT_BUG = f'{GIT_REPO_URL}/issues'

ABOUT = f'{GIT_REPO_URL}/blob/main/README.md'

APP_DESC = f"""
    This app leverages the *Mapper Algorithm* from Topological Data Analysis (TDA) to provide an efficient and intuitive way to gain insights from your datasets.

    For more details: 
    **{GIT_REPO_URL}**.
    """

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

# VD_ are reusable default values for widgets

VD_SEED = 42

VD_3D = False

# K_ are reusable keys for widgets

K_UPLOADER = 'key_uploader'

K_LENS_TYPE = 'key_lens_type'

K_LENS_PCA_N = 'key_lens_pca_n'

K_COVER_TYPE = 'key_cover_type'

K_COVER_BALL_RADIUS = 'key_cover_ball_radius'

K_COVER_BALL_METRIC_P = 'key_cover_metric_p'

K_COVER_CUBICAL_N = 'key_cover_cubical_n'

K_COVER_CUBICAL_OVERLAP = 'key_cover_cubical_overlap'

K_CLUSTERING_TYPE = 'key_clustering_type'

K_CLUSTERING_AGGLOMERATIVE_N = 'key_clustering_agglomerative_n'

K_ENABLE_3D = 'key_enable_3d'

K_SEED = 'key_seed'

K_DATA_SUMMARY = 'key_data_summary'

# S_ are reusable manually managed stored objects

S_RESULTS = 'stored_results'

# T_ are for call triggers

T_RENDER_MAPPER = True

T_DRAW_MAPPER = True


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
        self.df_summary = get_data_summary(self.df_X, self.df_y)

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


def mapper_warning(nodes_num):
    return f'''
        ⚠️ This graph contains {nodes_num} nodes, 
        which is more than the maximum allowed of {MAX_NODES}. 
        This may take time to display, make your browser run slow or either crash.
        Are you sure you want to proceed?
    '''


def lp_metric(p):
    return lambda x, y: np.linalg.norm(x - y, ord=p)


def data_caption(df_X, df_y):
    if df_X.empty:
        return 'No data source found'
    if df_y.empty:
        return f'{len(df_X)} instances, {len(df_X.columns)} features'
    return f'{len(df_X)} instances, {len(df_X.columns)} + {len(df_y.columns)} features'


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


def get_data_summary(df_X, df_y):
    df = pd.concat([get_sample(df_y), get_sample(df_X)], axis=1)
    df_hist = pd.DataFrame({x: df[x].value_counts(bins=V_DATA_SUMMARY_BINS, sort=False).values for x in df.columns}).T
    df_summary = pd.DataFrame({
        V_DATA_SUMMARY_FEAT: df.columns,
        V_DATA_SUMMARY_HIST: df_hist.values.tolist()
    })
    df_summary[V_DATA_SUMMARY_COLOR] = False
    return df_summary


@st.cache_data
def get_sample(df: pd.DataFrame, frac=SAMPLE_FRAC, max_n=MAX_SAMPLES, rand=42):
    if frac * len(df) > max_n:
        return df.sample(n=max_n, random_state=rand)
    return df.sample(frac=frac, random_state=rand)


@st.cache_data
def get_data_example(example):
    X, y = pd.DataFrame(), pd.DataFrame()
    if example == 'digits':
        X, y = load_digits(return_X_y=True, as_frame=True)
    elif example == 'iris':
        X, y = load_iris(return_X_y=True, as_frame=True)
    return fix_data(X), fix_data(y)


@st.cache_data
def get_data_openml(source):
    X, y = fetch_openml(source, return_X_y=True, as_frame=True)
    return fix_data(X), fix_data(y)


@st.cache_data
def get_data_csv(upload):
    df = pd.read_csv(upload)
    return fix_data(df), pd.DataFrame()


def load_data_openml(source):
    st.session_state[S_RESULTS].clear()
    X, y = get_data_openml(source)
    st.session_state[S_RESULTS].set_df(X, y)


def load_data_csv():
    source = st.session_state[K_UPLOADER]
    st.session_state[S_RESULTS].clear()
    X, y = get_data_csv(source)
    st.session_state[S_RESULTS].set_df(X, y)


def load_data_example(source):
    st.session_state[S_RESULTS].clear()
    X, y = get_data_example(source)
    st.session_state[S_RESULTS].set_df(X, y)


def add_download_graph():
    df_X = st.session_state[S_RESULTS]
    if df_X is None:
        return
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    mapper_adj = {} if mapper_graph is None else adjacency_data(mapper_graph)
    mapper_json = json.dumps(mapper_adj, default=int)
    st.download_button(
        f'📥 Download Graph',
        data=get_gzip_bytes(mapper_json),
        disabled=mapper_graph is None,
        file_name=f'mapper_graph_{int(time.time())}.json.gzip')
    if mapper_graph is None:
        return
    nodes_num = mapper_graph.number_of_nodes()
    edges_num = mapper_graph.number_of_edges()
    st.caption(f'{nodes_num} nodes, {edges_num} edges')


def add_data_source_csv():
    uploader = st.file_uploader(
        'Upload CSV',
        label_visibility='collapsed',
        on_change=load_data_csv,
        key=K_UPLOADER)


def add_data_source_example():
    example = st.selectbox(
        'Dataset Name',
        options=['digits', 'iris'])
    load = st.button('Load Example')
    if load:
        load_data_example(example)


def add_data_source_openml():
    name = st.text_input(
        'Dataset Name',
        placeholder='Dataset Name')
    load = st.button('Fetch from OpenML')
    if load:
        load_data_openml(name)


def add_data_source():
    source = st.selectbox(
        'Source',
        options=['Example', 'OpenML', 'CSV'])
    if source == 'OpenML':
        add_data_source_openml()
    elif source == 'CSV':
        add_data_source_csv()
    elif source == 'Example':
        add_data_source_example()


def add_lens_settings():
    lens_type = st.selectbox(
        'Type',
        options=[V_LENS_IDENTITY, V_LENS_PCA],
        key=K_LENS_TYPE)
    if lens_type == V_LENS_PCA:
            st.number_input(
                'Components',
                value=1,
                min_value=1,
                key=K_LENS_PCA_N)


def add_cover_settings():
    cover_type = st.selectbox(
        'Type',
        options=[V_COVER_BALL, V_COVER_CUBICAL, V_COVER_TRIVIAL],
        key=K_COVER_TYPE)
    if cover_type == V_COVER_BALL:
        st.number_input(
            'Radius',
            value=100.0,
            min_value=0.0,
            key=K_COVER_BALL_RADIUS)
        st.number_input(
            'Lp Metric',
            value=2,
            min_value=1,
            key=K_COVER_BALL_METRIC_P)
    elif cover_type == V_COVER_CUBICAL:
        st.number_input(
            'Intervals',
            value=2,
            min_value=0,
            key=K_COVER_CUBICAL_N)
        st.number_input(
            'Overlap Fraction',
            value=0.10,
            min_value=0.0,
            max_value=1.0,
            key=K_COVER_CUBICAL_OVERLAP)


def add_clustering_settings():
    clustering_type = st.selectbox(
        'Type',
        options=[V_CLUSTERING_TRIVIAL, V_CLUSTERING_AGGLOMERATIVE],
        key=K_CLUSTERING_TYPE)
    if clustering_type == V_CLUSTERING_AGGLOMERATIVE:
        st.number_input(
            'Clusters',
            value=2,
            min_value=1,
            key=K_CLUSTERING_AGGLOMERATIVE_N)


def add_mapper_settings():
    df_X = st.session_state[S_RESULTS].df_X
    st.markdown('### ⚙️ Settings')
    with st.expander('Lens'):
        add_lens_settings()
    with st.expander('🌐 Cover'):
        add_cover_settings()
    with st.expander('🧮 Clustering'):
        add_clustering_settings()
    run = st.button(
        '🚀 Run Mapper',
        type='primary',
        disabled=df_X is None)
    if run:
        with st.spinner('⏳ Computing Mapper...'):
            compute_mapper()
    add_download_graph()


def get_lens_func():
    lens_type = st.session_state.get(K_LENS_TYPE, V_LENS_IDENTITY)
    if lens_type == V_LENS_IDENTITY:
        return lambda x: x
    elif lens_type == V_LENS_PCA:
        n = st.session_state.get(K_LENS_PCA_N, 1)
        return lambda x: PCA(n).fit_transform(x)
    else:
        return lambda x: x


def get_cover_algo():
    cover_type = st.session_state.get(K_COVER_TYPE, V_COVER_TRIVIAL)
    if cover_type == V_COVER_TRIVIAL:
        return TrivialCover()
    elif cover_type == V_COVER_BALL:
        radius = st.session_state.get(K_COVER_BALL_RADIUS, 100.0)
        p = st.session_state.get(K_COVER_BALL_METRIC_P, 2)
        return BallCover(radius=radius, metric=lp_metric(p))
    elif cover_type == V_COVER_CUBICAL:
        n = st.session_state.get(K_COVER_CUBICAL_N, 10)
        p = st.session_state.get(K_COVER_CUBICAL_OVERLAP, 0.5)
        return CubicalCover(n_intervals=n, overlap_frac=p)


def get_clustering_algo():
    clustering_type = st.session_state.get(K_CLUSTERING_TYPE, None)
    if clustering_type == V_CLUSTERING_TRIVIAL:
        return TrivialClustering()
    if clustering_type == V_CLUSTERING_AGGLOMERATIVE:
        n = st.session_state.get(K_CLUSTERING_AGGLOMERATIVE_N, 2)
        return AgglomerativeClustering(n_clusters=n)


def compute_mapper():
    df_X = st.session_state[S_RESULTS].df_X
    if df_X is None:
        return
    X = st.session_state[S_RESULTS].X
    lens_func = get_lens_func()
    lens = lens_func(X)
    st.session_state[S_RESULTS].set_lens(lens)
    mapper_algo = MapperAlgorithm(
        cover=get_cover_algo(),
        clustering=FailSafeClustering(
            clustering=get_clustering_algo(),
            verbose=False))
    mapper_graph = mapper_algo.fit_transform(X, lens)
    st.session_state[S_RESULTS].clear_mapper()
    st.session_state[S_RESULTS].set_mapper_graph(mapper_graph)
    render_mapper()


def render_mapper():
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    if mapper_graph is None:
        return
    nodes_num = mapper_graph.number_of_nodes()
    if nodes_num > MAX_NODES:
        st.warning(mapper_warning(nodes_num))
        render = st.button(MAPPER_PROCEED)
        if render:
            render_mapper_proceed()
    else:
        render_mapper_proceed()


def render_mapper_proceed():
    X = st.session_state[S_RESULTS].X
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    seed = st.session_state.get(K_SEED, VD_SEED)
    enable_3d = st.session_state.get(K_ENABLE_3D, VD_3D)
    mapper_plot = MapperLayoutInteractive(
        mapper_graph,
        dim=3 if enable_3d else 2,
        height=450,
        width=450,
        colors=X,
        seed=seed)
    st.session_state[S_RESULTS].set_mapper_plot(mapper_plot)
    st.session_state[T_RENDER_MAPPER] = False
    draw_mapper()


def draw_mapper():
    mapper_plot = st.session_state[S_RESULTS].mapper_plot
    if mapper_plot is None:
        return
    colors = get_colors_data_summary()
    seed = st.session_state[K_SEED]
    mapper_plot.update(colors=colors, seed=seed)
    mapper_fig = mapper_plot.plot()
    mapper_fig.update_layout(uirevision='constant')
    st.session_state['mapper_fig'] = mapper_fig
    st.session_state[T_DRAW_MAPPER] = False


def get_colors_data_summary():
    df_X = st.session_state[S_RESULTS].df_X
    if df_X is None:
        df_X = pd.DataFrame()
    df_y = st.session_state[S_RESULTS].df_y
    if df_y is None:
        df_y = pd.DataFrame()
    df_summary = st.session_state[S_RESULTS].df_summary
    summary = st.session_state[K_DATA_SUMMARY]
    edited = summary['edited_rows'].items()
    rows = [k for k, v in edited if v.get(V_DATA_SUMMARY_COLOR, False)]
    cols = df_summary[V_DATA_SUMMARY_FEAT].iloc[rows].values
    df_cols = []
    for c in cols:
        if c in df_X.columns:
            df_cols.append(df_X[c])
        elif c in df_y.columns:
            df_cols.append(df_y[c])
    if not df_cols:
        lens = st.session_state[S_RESULTS].lens
        return lens
    colors = pd.concat(df_cols, axis=1).to_numpy()
    return colors


def add_data_tools():
    df_X = st.session_state[S_RESULTS].df_X
    if df_X is None:
        return
    df_y = st.session_state[S_RESULTS].df_y
    df_summary = st.session_state[S_RESULTS].df_summary
    if df_summary is None:
        df_summary = pd.DataFrame()
    
    def _trigger_draw_mapper():
        st.session_state[T_DRAW_MAPPER] = True
    st.data_editor(
        df_summary,
        height=250,
        hide_index=True,
        disabled=(c for c in df_summary.columns if c != V_DATA_SUMMARY_COLOR),
        use_container_width=True,
        column_config={
            V_DATA_SUMMARY_HIST: st.column_config.BarChartColumn(
                width='small'),
        },
        key=K_DATA_SUMMARY,
        on_change=_trigger_draw_mapper)


def add_plot_setting():
    def _trigger_draw_mapper():
        st.session_state[T_DRAW_MAPPER] = True
    def _trigger_render_mapper():
        st.session_state[T_RENDER_MAPPER] = True
    seed = st.number_input(
        'Seed',
        value=VD_SEED,
        key=K_SEED,
        on_change=_trigger_draw_mapper)
    st.toggle(
        'Enable 3D',
        value=VD_3D,
        key=K_ENABLE_3D,
        on_change=_trigger_render_mapper)
    mapper_graph = st.session_state[S_RESULTS].mapper_graph


def add_graph_plot():
    if 'mapper_fig' not in st.session_state:
        return
    mapper_graph = st.session_state[S_RESULTS].mapper_graph
    if mapper_graph is None:
        return
    mapper_fig = st.session_state['mapper_fig']
    st.plotly_chart(
        mapper_fig,
        use_container_width=True,
        config={'scrollZoom': True})


def add_data():
    st.markdown('### 📊 Data')
    df_X = st.session_state[S_RESULTS].df_X
    df_y = st.session_state[S_RESULTS].df_y
    add_data_source()
    df_X = st.session_state[S_RESULTS].df_X
    df_y = st.session_state[S_RESULTS].df_y
    if df_X is None:
        return
    cap = data_caption(df_X, df_y)
    with st.expander(cap):
        df_all = pd.concat([get_sample(df_y, frac=1.0), get_sample(df_X, frac=1.0)], axis=1)
        st.dataframe(df_all, height=300)


def add_rendering():
    st.markdown('### 🎨 Rendering')
    pl_col_0, pl_col_1 = st.columns([2, 4])
    with pl_col_0:
        add_data_tools()
        add_plot_setting()
    with pl_col_1:
        if st.session_state.get(T_RENDER_MAPPER, True):
            render_mapper()
        if st.session_state.get(T_DRAW_MAPPER, True):
            draw_mapper()
        add_graph_plot()
    

def main():
    st.set_page_config(
        page_icon='app/logo_icon.png',
        page_title='tda-mapper',
        menu_items={
            'Report a bug': REPORT_BUG,
            'About': ABOUT
        })
    try:
        logo_hori = open('app/logo_hori.png')
        with st.sidebar:
            st.image(logo_hori, use_column_width=True)
            st.markdown('###')
        st.image(logo_hori, use_column_width=True)
        logo_hori.close()
    except Exception:
        title = '🔮 TDA Mapper App'
        with st.sidebar:
            st.markdown(f'# {title}')
        st.markdown(f'# {title}')
    finally:
        with st.sidebar:
            st.markdown('#')
        st.markdown('#')

    with st.sidebar:
        st.markdown(APP_DESC)
    if S_RESULTS not in st.session_state:
        st.session_state[S_RESULTS] = Results()
    st.markdown('#')
    add_data()
    st.markdown('#')
    add_mapper_settings()
    st.markdown('#')
    add_rendering()
    st.markdown(f'''
        ---
        If you found this app useful please consider leaving a :star: on {GIT_REPO_URL}
    ''')


main()
