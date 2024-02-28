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


def mapper_warning(nodes_num):
    return f'''
        ⚠️ This graph contains {nodes_num} nodes, 
        which is more than the maximum allowed of {MAX_NODES}. 
        This may take time to display, make your browser run slow or either crash.
        Are you sure you want to proceed?
    '''


def data_caption(df_X, df_y):
    caption = f'{len(df_X)} samples, {len(df_X.columns)} source features'
    if df_y is not None:
        caption = f'''
            {len(df_X)} samples, 
            {len(df_X.columns)} source features, 
            {len(df_y.columns)} target features
        '''
    return caption


def data_concat_head(df_X, df_y):
    if df_y is None:
        return df_X.head()
    return pd.concat([df_X.head(), df_y.head()], axis=1)


def fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include='number')
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


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


def lp_metric(p):
    return lambda x, y: np.linalg.norm(x - y, ord=p)


def gzip_bytes(string, encoding='utf-8'):
    fileobj = io.BytesIO()
    gzf = gzip.GzipFile(fileobj=fileobj, mode='wb', compresslevel=6)
    gzf.write(string.encode(encoding))
    gzf.close()
    return fileobj.getvalue()


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


def get_data():
    st.write('## 📊 Data Source')
    source = st.radio(
        'Data Source',
        options=['Example', 'OpenML', 'CSV'],
        horizontal=True,
        label_visibility='collapsed')
    if source == 'OpenML':
        name = st.text_input(
            'Dataset Name',
            label_visibility='collapsed',
            placeholder='Dataset Name')
        st.button(
            'Fetch from OpenML',
            on_click=load_data_openml,
            args=(name,))
    elif source == 'CSV':
        st.file_uploader(
            'Upload CSV',
            label_visibility='collapsed',
            on_change=load_data_csv)
    elif source == 'Example':
        example = st.selectbox(
            'Example',
            options=['digits', 'iris'],
            label_visibility='collapsed')
        st.button(
            'Load Example',
            on_click=load_data_example,
            args=(example,))


def get_mapper_lens():
    lens_func = lambda x: x
    lens_type = st.selectbox(
        'Lens',
        options=['Identity', 'PCA'],
        label_visibility='collapsed')
    if lens_type == 'Identity':
        lens_func = lambda x: x
    if lens_type == 'PCA':
        lens_pca_n = st.number_input(
            'PCA components',
            value=1,
            min_value=1)
        lens_func = lambda x: PCA(lens_pca_n).fit_transform(x)
    return lens_func


def get_mapper_cover():
    mapper_cover = TrivialCover()
    cover_type = st.selectbox(
        'Cover',
        options=['Ball', 'Cubical'],
        label_visibility='collapsed')
    if cover_type == 'Ball':
        cover_ball_radius = st.number_input(
            'Ball radius',
            value=100.0,
            min_value=0.0)
        cover_ball_metric_p = st.number_input(
            'Lp metric',
            value=2,
            min_value=1)
        mapper_cover = BallCover(
            radius=cover_ball_radius,
            metric=lp_metric(cover_ball_metric_p))
    elif cover_type == 'Cubical':
        cover_cubical_n = st.number_input(
            'intervals',
            value=2,
            min_value=0)
        cover_cubical_overlap = st.number_input(
            'overlap',
            value=0.10,
            min_value=0.0,
            max_value=1.0)
        mapper_cover = CubicalCover(
            n_intervals=cover_cubical_n,
            overlap_frac=cover_cubical_overlap)
    return mapper_cover


def get_mapper_clustering():
    mapper_clustering = TrivialClustering()
    clustering_type = st.selectbox(
        'Clustering',
        options=['Trivial', 'Agglomerative'],
        label_visibility='collapsed')
    if clustering_type == 'Trivial':
        mapper_clustering = TrivialClustering()
    if clustering_type == 'Agglomerative':
        clustering_agglomerative_n = st.number_input(
            'clusters',
            value=2,
            min_value=1)
        mapper_clustering = AgglomerativeClustering(
            n_clusters=int(clustering_agglomerative_n))
    return mapper_clustering


def compute_mapper(X, lens, mapper):
    mapper_graph = mapper.fit_transform(X, lens)
    st.toast(MAPPER_COMPUTED, icon='🎉')
    st.session_state['X'] = X
    st.session_state['lens'] = lens
    clear_session_mapper()
    st.session_state['mapper_graph'] = mapper_graph


def run_mapper():
    st.write('## 🔮 Mapper Settings')
    with st.expander('🔎 Lens'):
        mapper_lens = get_mapper_lens()
    with st.expander('🌐 Cover'):
        mapper_cover = get_mapper_cover()
    with st.expander('🧮 Clustering'):
        mapper_clustering = get_mapper_clustering()
    mapper = MapperAlgorithm(
        cover=mapper_cover,
        clustering=FailSafeClustering(mapper_clustering))
    df_X = st.session_state.get('df_X', pd.DataFrame())
    X = df_X.to_numpy()
    lens = mapper_lens(X)
    st.button(
        '✨ Run',
        use_container_width=True,
        disabled=df_X.empty,
        on_click=compute_mapper,
        args=(X, lens, mapper,))


def render_graph(X, mapper_graph, colors):
    nodes_num = mapper_graph.number_of_nodes()
    edges_num = mapper_graph.number_of_edges()
    st.caption(f'{nodes_num} nodes, {edges_num} edges')
    enable_3d = st.toggle('Enable 3d', value=True)
    dim = 3 if enable_3d else 2
    if nodes_num > MAX_NODES:
        st.warning(mapper_warning(nodes_num))
        proceed = st.button(
            MAPPER_PROCEED,
            type='primary')
        if proceed:
            plot_graph(X, mapper_graph, colors, dim)
    else:
        plot_graph(X, mapper_graph, colors, dim)


def plot_graph(X, mapper_graph, colors, dim, seed=42):
    mapper_plot = MapperPlot(X, mapper_graph,
        colors=colors,
        dim=dim,
        seed=seed)
    mapper_fig = mapper_plot.plot(
        backend='plotly',
        height=600,
        width=800)
    st.session_state['mapper_fig'] = mapper_fig


def display_data_source(df_X, df_y):
    st.caption(data_caption(df_X, df_y),
        help=DATA_INFO)
    st.dataframe(data_concat_head(df_X, df_y),
        hide_index=True,
        height=100)


def download_graph(mapper_graph):
    mapper_adj = nx.readwrite.json_graph.adjacency_data(mapper_graph)
    mapper_json = json.dumps(mapper_adj)
    st.download_button(
        '📥 Download Mapper Graph',
        data=gzip_bytes(mapper_json),
        file_name=f'mapper_graph_{int(time.time())}.json.gzip')


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
    with st.sidebar:
        get_data()
        run_mapper()
    if 'df_X' in st.session_state:
        df_X = st.session_state['df_X']
        df_y = st.session_state.get('df_y', None)
        display_data_source(df_X, df_y)
    else:
        st.write(DATA_HELP)
    if 'mapper_graph' in st.session_state:
        mapper_graph = st.session_state['mapper_graph']
        df_X = st.session_state['df_X']
        lens = st.session_state['lens']
        df_y = st.session_state.get('df_y', None)
        colors = lens if df_y is None else df_y.to_numpy()
        render_graph(df_X, mapper_graph, colors)
    else:
        st.write(MAPPER_HELP)
    if 'mapper_fig' in st.session_state:
        mapper_graph = st.session_state['mapper_graph']
        mapper_fig = st.session_state['mapper_fig']
        st.plotly_chart(mapper_fig, use_container_width=True)
        download_graph(mapper_graph)


main()
