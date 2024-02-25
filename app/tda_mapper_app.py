import streamlit as st
import pandas as pd

import numpy as np

from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover, BallCover
from tdamapper.clustering import TrivialClustering, FailSafeClustering
from tdamapper.plot import MapperPlot

MAX_NODES = 1000


@st.cache_data
def load_data_example():
    X, y = load_digits(return_X_y=True, as_frame=True)
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    return df_X, df_y


@st.cache_data
def load_data_openml(source):
    X, y = fetch_openml(source, return_X_y=True, as_frame=True)
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    return df_X, df_y


@st.cache_data
def load_data_csv(upload):
    df_X = pd.read_csv(upload)
    return df_X, df_X


def lp_metric(p):
    return lambda x, y: np.linalg.norm(x - y, ord=p)


def get_data():
    source = st.radio('Data Source', options=['CSV', 'OpenML', 'Example'], horizontal=True, label_visibility='collapsed')
    if source == 'OpenML':
        dataset_name = st.text_input('Dataset Name', label_visibility='collapsed', placeholder='Dataset Name')
        fetch_button = st.button('Fetch from OpenML')
        if fetch_button:
            df_X, df_y = load_data_openml(dataset_name)
            st.session_state['df_X'] = df_X
            st.session_state['df_y'] = df_y
    elif source == 'CSV':
        upload = st.file_uploader('Upload CSV', label_visibility='collapsed')
        if upload:
            df_X = pd.read_csv(upload)
            st.session_state['df_X'] = df_X
    elif source == 'Example':
        example = st.selectbox('Example', options=['digits', 'iris'])
        load_button = st.button('Load Example')
        if load_button:
            if example == 'digits':
                df_X, df_y = load_digits(return_X_y=True, as_frame=True)
                st.session_state['df_X'] = df_X
                st.session_state['df_y'] = df_y
            elif example == 'iris':
                df_X, df_y = load_iris(return_X_y=True, as_frame=True)
                st.session_state['df_X'] = df_X
                st.session_state['df_y'] = df_y


def get_mapper_lens():
    with st.expander('🔎 Lens'):
        lens_type = st.selectbox('Lens', options=['Identity', 'PCA'], label_visibility='collapsed')
        if lens_type == 'Identity':
            lens_func = lambda x: x
        if lens_type == 'PCA':
            lens_pca_n = st.number_input('PCA components', value=1, min_value=1)
            lens_func = lambda x: PCA(lens_pca_n).fit_transform(x)
    return lens_func


def get_mapper_cover():
    with st.expander('🌐 Cover'):
        cover_type = st.selectbox('Cover', options=['Ball', 'Cubical'], label_visibility='collapsed')
        if cover_type == 'Ball':
            cover_ball_radius = st.number_input('Ball radius', value=100.0, min_value=0.0)
            cover_ball_metric_p = st.number_input('Lp metric', value=2, min_value=1)
            mapper_cover = BallCover(
                radius=cover_ball_radius, 
                metric=lp_metric(cover_ball_metric_p))
        elif cover_type == 'Cubical':
            cover_cubical_n = st.number_input('intervals', value=2, min_value=0)
            cover_cubical_overlap = st.number_input('overlap', value=0.10, min_value=0.0, max_value=1.0)
            mapper_cover = CubicalCover(
                n_intervals=cover_cubical_n, 
                overlap_frac=cover_cubical_overlap)
    return mapper_cover


def get_mapper_clustering():
    with st.expander('🧮 Clustering'):
        clustering_type = st.selectbox('Clustering', options=['Trivial', 'Agglomerative'], label_visibility='collapsed')
        if clustering_type == 'Trivial':
            mapper_clustering = TrivialClustering()
        if clustering_type == 'Agglomerative':
            clustering_agglomerative_n = st.number_input('clusters', value=2, min_value=1)
            mapper_clustering = AgglomerativeClustering(n_clusters=clustering_agglomerative_n)
    return mapper_clustering


def run_mapper():
    mapper_lens = get_mapper_lens()
    mapper_cover = get_mapper_cover()
    mapper_clustering = get_mapper_clustering()
    mapper = MapperAlgorithm(
        cover=mapper_cover,
        clustering=FailSafeClustering(mapper_clustering))
    df_X = st.session_state.get('df_X')
    compute = st.button('🚀 Run', use_container_width=True, disabled=df_X is None)
    if compute:
        X = df_X.to_numpy()
        lens = mapper_lens(X)
        mapper_graph = mapper.fit_transform(X, lens)
        st.toast('Mapper graph computed!', icon='🎉')
        st.session_state['X'] = X
        st.session_state['lens'] = lens
        st.session_state['mapper_graph'] = mapper_graph


def render_graph():
    X = st.session_state['X']
    lens = st.session_state['lens']
    if 'df_y' in st.session_state:
        df_y = st.session_state['df_y']
        color = df_y.to_numpy
    else:
        color = lens
    mapper_graph = st.session_state['mapper_graph']
    nodes_num = mapper_graph.number_of_nodes()
    edges_num = mapper_graph.number_of_edges()
    st.caption(f'{nodes_num} nodes, {edges_num} edges')
    enable_3d = st.toggle('Enable 3d')
    dim = 3 if enable_3d else 2
    if nodes_num > MAX_NODES:
        st.warning(f'This graph contains {nodes_num} nodes, which is more than the maximum allowed of {MAX_NODES}. This may take time to display or make your browser run slow.')
        show_anyway = st.button('Show me the damn graph!')
        if show_anyway:
            draw_graph(X, lens, color, mapper_graph, dim)
    else:
        draw_graph(X, lens, color, mapper_graph, dim)


def draw_graph(X, lens, color, mapper_graph, dim):
    mapper_plot = MapperPlot(X, mapper_graph, colors=lens, dim=dim, seed=42)
    mapper_fig = mapper_plot.plot(backend='plotly', height=500, width=500)
    st.plotly_chart(mapper_fig, use_container_width=True)


def main():
    st.set_page_config(layout='wide', page_icon='🚀', menu_items={
        'Report a bug': 'https://github.com/lucasimi/tda-mapper-python/issues',
        'About': 'https://github.com/lucasimi/tda-mapper-python/README.md'
    })
    st.title('tda-mapper-demo')
    tab_input, tab_output = st.tabs(['📊 Data Source', '🔥 Mapper Graph'])
    with st.sidebar:
        st.write('## 📊 Data Source')
        get_data()
        st.write('## ⚙️ Mapper Settings')
        run_mapper()
    if 'df_X' in st.session_state:
        df_X = st.session_state['df_X']
        with tab_input:
            st.caption(f'{len(df_X)} samples, {len(df_X.columns)} features')
            st.dataframe(df_X)
    else:
        with tab_input:
            st.write(
                '''
                Select you Data Source: submit a dataset as a csv file or use a publicly available one from [OpenML](https://www.openml.org/search?type=data&sort=runs&status=active).
                ''')
    if 'mapper_graph' in st.session_state:
        with tab_output:
            render_graph()
    else:
        with tab_output:
            st.write(
                '''
                Experiment with Mapper Settings and hit Run
                ''')


main()
