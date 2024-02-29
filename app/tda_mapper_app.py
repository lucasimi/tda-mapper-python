import json
import time
import io
import gzip

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover, BallCover, TrivialCover
from tdamapper.clustering import TrivialClustering, FailSafeClustering
from tdamapper.plot import MapperPlot


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

DATA_INFO = 'Non-numeric and NaN features are dropped. NaN rows are replaced by mean'

MAPPER_HELP = "Experiment with Mapper Settings and hit Run when you're ready!"

MAPPER_PROCEED = '💣 Go on and show me the damn graph!'

MAPPER_COMPUTED = 'Mapper Graph Computed!'

GIT_REPO_URL = 'https://github.com/lucasimi/tda-mapper-python'

REPORT_BUG = f'{GIT_REPO_URL}/issues'

ABOUT = f'{GIT_REPO_URL}/README.md'

LENS_IDENTITY = 'Identity'

LENS_PCA = 'PCA'

COVER_TRIVIAL = 'Trivial'

COVER_BALL = 'Ball'

COVER_CUBICAL = 'Cubical'

CLUSTERING_TRIVIAL = 'Trivial'

CLUSTERING_AGGLOMERATIVE = 'Agglomerative'

PLOT_COLOR_LENS = 'lens'

KEY_LENS_TYPE = 'key_lens_type'

KEY_LENS_PCA_N = 'key_lens_pca_n'

KEY_COVER_TYPE = 'key_cover_type'

KEY_COVER_BALL_RADIUS = 'key_cover_ball_radius'

KEY_COVER_BALL_METRIC_P = 'key_cover_metric_p'

KEY_COVER_CUBICAL_N = 'key_cover_cubical_n'

KEY_COVER_CUBICAL_OVERLAP = 'key_cover_cubical_overlap'

KEY_CLUSTERING_TYPE = 'key_clustering_type'

KEY_CLUSTERING_AGGLOMERATIVE_N = 'key_clustering_agglomerative_n'

KEY_ENABLE_3D = 'key_enable_3d'

KEY_SEED = 'key_seed'

KEY_PLOT_COLOR = 'key_plot_color'

DEFAULT_SEED = 42

DEFAULT_3D = True


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
    if df_y.empty:
        return f'{len(df_X)} instance, {len(df_X.columns)} features'
    return f'''
        {len(df_X)} instances, {len(df_X.columns)} + {len(df_y.columns)} features
    '''


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
def get_sample(df, frac=0.1):
    return df.sample(frac=frac)


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
    return fix_data(df)


def load_data_openml(source):
    clear_session_data()
    df_X, df_y = get_data_openml(source)
    st.session_state['df_X'] = df_X
    st.session_state['df_y'] = df_y


def load_data_csv(source):
    clear_session_data()
    df_X, df_y = get_data_csv(source)
    st.session_state['df_X'] = df_X
    st.session_state['df_y'] = df_y


def load_data_example(source):
    clear_session_data()
    df_X, df_y = get_data_example(source)
    st.session_state['df_X'] = df_X
    st.session_state['df_y'] = df_y


def clear_session_source():
    st.session_state.pop('df_X', None)
    st.session_state.pop('df_y', None)
    st.session_state.pop('X', None)
    st.session_state.pop('lens', None)


def clear_session_mapper():
    st.session_state.pop('mapper_graph', None)
    st.session_state.pop('mapper_fig', None)


def clear_session_data():
    clear_session_source()
    clear_session_mapper()


def get_data_summary(df):
    df_hist = pd.DataFrame({x: df[x].value_counts(bins=10, sort=False).values for x in df.columns}).T
    df_summary = pd.DataFrame({
        'feature': df.columns,
        'hist': df_hist.values.tolist()
    })
    return df_summary


def download_graph(mapper_graph):
    mapper_adj = nx.readwrite.json_graph.adjacency_data(mapper_graph)
    mapper_json = json.dumps(mapper_adj)
    st.download_button('📥 Download Mapper Graph',
        data=get_gzip_bytes(mapper_json),
        use_container_width=True,
        file_name=f'mapper_graph_{int(time.time())}.json.gzip')


def add_data_source_csv():
    st.file_uploader('Upload CSV',
        label_visibility='collapsed',
        on_change=load_data_csv)


def add_data_source_example():
    example = st.selectbox('Example',
        options=['digits', 'iris'],
        label_visibility='collapsed')
    st.button('Load Example',
        on_click=load_data_example,
        args=(example,))


def add_data_source_openml():
    name = st.text_input('Dataset Name',
        label_visibility='collapsed',
        placeholder='Dataset Name')
    st.button('Fetch from OpenML',
        on_click=load_data_openml,
        args=(name,))


def add_data_source():
    st.write('## 📊 Data Source')
    source = st.radio('Data Source',
        options=['Example', 'OpenML', 'CSV'],
        horizontal=True,
        label_visibility='collapsed')
    if source == 'OpenML':
        add_data_source_openml()
    elif source == 'CSV':
        add_data_source_csv()
    elif source == 'Example':
        add_data_source_example()


def add_mapper_settings():
    st.write('## 🔮 Mapper Settings')
    with st.expander('🔎 Lens'):
        add_lens_settings()
    with st.expander('🌐 Cover'):
        add_cover_settings()
    with st.expander('🧮 Clustering'):
        add_clustering_settings()
    st.button('✨ Run',
        use_container_width=True,
        disabled='df_X' not in st.session_state,
        on_click=compute_mapper)


def add_lens_settings():
    lens_type = st.selectbox('Lens',
        options=[LENS_IDENTITY, LENS_PCA],
        label_visibility='collapsed',
        key=KEY_LENS_TYPE)
    if lens_type == LENS_PCA:
        st.number_input('PCA components',
            value=1,
            min_value=1,
            key=KEY_LENS_PCA_N)


def get_lens_func():
    lens_type = st.session_state.get(KEY_LENS_TYPE, LENS_IDENTITY)
    if lens_type == LENS_IDENTITY:
        return lambda x: x
    elif lens_type == LENS_PCA:
        n = st.session_state.get(KEY_LENS_PCA_N, 1)
        return lambda x: PCA(n).fit_transform(x)
    else:
        return lambda x: x


def add_cover_settings():
    cover_type = st.selectbox('Cover',
        options=[COVER_BALL, COVER_CUBICAL, COVER_TRIVIAL],
        label_visibility='collapsed',
        key=KEY_COVER_TYPE)
    if cover_type == COVER_BALL:
        st.number_input('Ball radius',
            value=100.0,
            min_value=0.0,
            key=KEY_COVER_BALL_RADIUS)
        st.number_input('Lp metric',
            value=2,
            min_value=1,
            key=KEY_COVER_BALL_METRIC_P)
    elif cover_type == COVER_CUBICAL:
        st.number_input('intervals',
            value=2,
            min_value=0,
            key=KEY_COVER_CUBICAL_N)
        st.number_input('overlap',
            value=0.10,
            min_value=0.0,
            max_value=1.0,
            key=KEY_COVER_CUBICAL_OVERLAP)


def get_cover_algo():
    cover_type = st.session_state.get(KEY_COVER_TYPE, COVER_TRIVIAL)
    if cover_type == COVER_TRIVIAL:
        return TrivialCover()
    elif cover_type == COVER_BALL:
        radius = st.session_state.get(KEY_COVER_BALL_RADIUS, 100.0)
        p = st.session_state.get(KEY_COVER_BALL_METRIC_P, 2)
        return BallCover(radius=radius, metric=lp_metric(p))
    elif cover_type == COVER_CUBICAL:
        n = st.session_state.get(KEY_COVER_CUBICAL_N, 10)
        p = st.session_state.get(KEY_COVER_CUBICAL_OVERLAP, 0.5)
        return CubicalCover(n_intervals=n, overlap_frac=p)


def add_clustering_settings():
    clustering_type = st.selectbox('Clustering',
        options=[CLUSTERING_TRIVIAL, CLUSTERING_AGGLOMERATIVE],
        label_visibility='collapsed',
        key=KEY_CLUSTERING_TYPE)
    if clustering_type == CLUSTERING_AGGLOMERATIVE:
        st.number_input('clusters',
            value=2,
            min_value=1,
            key=KEY_CLUSTERING_AGGLOMERATIVE_N)


def get_clustering_algo():
    clustering_type = st.session_state.get(KEY_CLUSTERING_TYPE, None)
    if clustering_type == CLUSTERING_TRIVIAL:
        return TrivialClustering()
    if clustering_type == CLUSTERING_AGGLOMERATIVE:
        n = st.session_state.get(KEY_CLUSTERING_AGGLOMERATIVE_N, 2)
        return AgglomerativeClustering(n_clusters=n)


def compute_mapper():
    if 'df_X' not in st.session_state:
        return
    df_X = st.session_state['df_X']
    X = df_X.to_numpy()
    lens_func = get_lens_func()
    lens = lens_func(X)
    st.session_state['X'] = X
    st.session_state['lens'] = lens
    mapper_algo = MapperAlgorithm(
        cover=get_cover_algo(),
        clustering=FailSafeClustering(
            clustering=get_clustering_algo(),
            verbose=False))
    with st.spinner('Computing Mapper Graph'):
        mapper_graph = mapper_algo.fit_transform(X, lens)
    st.session_state['mapper_graph'] = mapper_graph
    render_mapper()


def render_mapper():
    if 'mapper_graph' not in st.session_state:
        return
    mapper_graph = st.session_state['mapper_graph']
    nodes_num = mapper_graph.number_of_nodes()
    if nodes_num > MAX_NODES:
        st.warning(mapper_warning(nodes_num))
        st.button(MAPPER_PROCEED,
            on_click=render_mapper_proceed)
    else:
        render_mapper_proceed()


def render_mapper_proceed():
    df_X = st.session_state.get('df_X', pd.DataFrame())
    df_y = st.session_state.get('df_y', pd.DataFrame())
    X = st.session_state.get('X', None)
    mapper_graph = st.session_state['mapper_graph']
    lens = st.session_state['lens']
    seed = st.session_state.get(KEY_SEED, DEFAULT_SEED)
    enable_3d = st.session_state.get(KEY_ENABLE_3D, DEFAULT_3D)
    plot_color = st.session_state.get(KEY_PLOT_COLOR, PLOT_COLOR_LENS)
    colors = lens
    if plot_color == PLOT_COLOR_LENS:
        colors = lens
    if plot_color in df_X.columns:
        colors = df_X[plot_color].to_numpy()
    if plot_color in df_y.columns:
        colors = df_y[plot_color].to_numpy()
    mapper_plot = MapperPlot(X, mapper_graph,
        colors=colors,
        dim=3 if enable_3d else 2,
        seed=seed)
    st.session_state['mapper_plot'] = mapper_plot
    draw_mapper()


def draw_mapper():
    if 'mapper_plot' not in st.session_state:
        return
    mapper_plot = st.session_state['mapper_plot']
    with st.spinner('Drawing Mapper Graph'):
        mapper_fig = mapper_plot.plot(
            backend='plotly',
            height=600,
            width=800)
    st.session_state['mapper_fig'] = mapper_fig


def add_data_summary():
    df_X = st.session_state.get('df_X', pd.DataFrame())
    if df_X.empty:
        return
    df_y = st.session_state.get('df_y', pd.DataFrame())
    df_all = pd.concat([get_sample(df_X), get_sample(df_y)], axis=1)
    st.caption(data_caption(df_X, df_y),
        help=DATA_INFO)
    df_summary = get_data_summary(df_all)
    st.dataframe(df_summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "hist": st.column_config.BarChartColumn(),
        })


def add_plot_tools():
    if 'mapper_graph' not in st.session_state:
        return
    df_X = st.session_state['df_X']
    df_y = st.session_state.get('df_y', pd.DataFrame())
    st.toggle('Enable 3d',
        value=DEFAULT_3D,
        on_change=render_mapper,
        key=KEY_ENABLE_3D)
    st.number_input('Seed',
        value=DEFAULT_SEED,
        on_change=render_mapper,
        key=KEY_SEED)
    cols = [PLOT_COLOR_LENS] + [x for x in df_X.columns] + [x for x in df_y.columns]
    st.selectbox('Plot color',
        options=cols,
        on_change=render_mapper,
        key=KEY_PLOT_COLOR)
    mapper_graph = st.session_state['mapper_graph']
    download_graph(mapper_graph)


def add_graph_plot():
    if 'mapper_fig' not in st.session_state:
        return
    mapper_graph = st.session_state['mapper_graph']
    nodes_num = mapper_graph.number_of_nodes()
    edges_num = mapper_graph.number_of_edges()
    mapper_fig = st.session_state['mapper_fig']
    st.caption(f'{nodes_num} nodes, {edges_num} edges')
    st.plotly_chart(mapper_fig,
        use_container_width=True)


def main():
    st.set_page_config(
        layout='wide',
        page_icon='🔮',
        page_title='tda-mapper-app',
        menu_items={
            'Report a bug': REPORT_BUG,
            'About': ABOUT
        })
    st.title('tda-mapper-app')
    with st.sidebar.container(border=False):
        add_data_source()
    with st.sidebar.container(border=False):
        add_mapper_settings()
    col_tools, col_graph = st.columns([2, 5])
    with col_tools:
        add_data_summary()
        add_plot_tools()
    with col_graph:
        add_graph_plot()


main()
