import streamlit as st
import pandas as pd

import numpy as np

from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from tdamapper.core import MapperAlgorithm
from tdamapper.cover import CubicalCover, BallCover, TrivialCover
from tdamapper.clustering import TrivialClustering, FailSafeClustering
from tdamapper.plot import MapperPlot

MAX_NODES = 1000

MAPPER_EMOJI = '🔮'


def fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include='number')
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


@st.cache_data
def load_data_example(example):
    X, y = pd.DataFrame(), pd.DataFrame()
    if example == 'digits':
        X, y = load_digits(return_X_y=True, as_frame=True)
    elif example == 'iris':
        X, y = load_iris(return_X_y=True, as_frame=True)
    return fix_data(X), fix_data(y)


@st.cache_data
def load_data_openml(source):
    X, y = fetch_openml(source, return_X_y=True, as_frame=True)
    return fix_data(X), fix_data(y)


@st.cache_data
def load_data_csv(upload):
    df = pd.read_csv(upload)
    return fix_data(df)


def lp_metric(p):
    return lambda x, y: np.linalg.norm(x - y, ord=p)


def get_data():
    source = st.radio('Data Source', options=['Example', 'OpenML', 'CSV'], horizontal=True, label_visibility='collapsed')
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
            df_X = load_data_csv(upload)
            st.session_state['df_X'] = df_X
    elif source == 'Example':
        example = st.selectbox('Example', options=['digits', 'iris'], label_visibility='collapsed')
        load_button = st.button('Load Example')
        if load_button:
            df_X, df_y = load_data_example(example)
            st.session_state['df_X'] = df_X
            st.session_state['df_y'] = df_y


def get_mapper_lens():
    lens_func = lambda x: x
    with st.expander('🔎 Lens'):
        lens_type = st.selectbox('Lens', options=['Identity', 'PCA'], label_visibility='collapsed')
        if lens_type == 'Identity':
            lens_func = lambda x: x
        if lens_type == 'PCA':
            lens_pca_n = st.number_input('PCA components', value=1, min_value=1)
            lens_func = lambda x: PCA(lens_pca_n).fit_transform(x)
    return lens_func


def get_mapper_cover():
    mapper_cover = TrivialCover()
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
    mapper_clustering = TrivialClustering()
    with st.expander('🧮 Clustering'):
        clustering_type = st.selectbox('Clustering', options=['Trivial', 'Agglomerative'], label_visibility='collapsed')
        if clustering_type == 'Trivial':
            mapper_clustering = TrivialClustering()
        if clustering_type == 'Agglomerative':
            clustering_agglomerative_n = st.number_input('clusters', value=2, min_value=1)
            mapper_clustering = AgglomerativeClustering(n_clusters=int(clustering_agglomerative_n))
    return mapper_clustering


def run_mapper():
    mapper_lens = get_mapper_lens()
    mapper_cover = get_mapper_cover()
    mapper_clustering = get_mapper_clustering()
    mapper = MapperAlgorithm(
        cover=mapper_cover,
        clustering=FailSafeClustering(mapper_clustering))
    df_X = st.session_state.get('df_X')
    compute = st.button('✨ Run', use_container_width=True, disabled=df_X is None)
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
        colors = df_y.to_numpy()
    else:
        colors = lens
    mapper_graph = st.session_state['mapper_graph']
    nodes_num = mapper_graph.number_of_nodes()
    edges_num = mapper_graph.number_of_edges()
    st.caption(f'{nodes_num} nodes, {edges_num} edges')
    #enable_3d = st.toggle('Enable 3d')
    enable_3d=True
    dim = 3 if enable_3d else 2
    if nodes_num > MAX_NODES:
        st.warning(f'''
            ⚠️ This graph contains {nodes_num} nodes, 
            which is more than the maximum allowed of {MAX_NODES}. 
            This may take time to display, make your browser run slow or either crash.
            Are you sure you want to proceed?
            ''')
        show_anyway = st.button('💣 Go on and show me the damn graph!', type='primary')
        if show_anyway:
            draw_graph(X, mapper_graph, colors, dim)
    else:
        draw_graph(X, mapper_graph, colors, dim)


def draw_graph(X, mapper_graph, colors, dim, seed=42):
    mapper_plot = MapperPlot(X, mapper_graph, colors=colors, dim=dim, seed=seed)
    mapper_fig = mapper_plot.plot(backend='plotly', height=600, width=800)
    st.plotly_chart(mapper_fig, use_container_width=True)


def display_data_source():
    if 'df_X' in st.session_state:
        df_X = st.session_state['df_X']
        df_all = df_X.head()
        caption = f'{len(df_X)} samples, {len(df_X.columns)} source features'
        help = 'Non-numeric and NaN features are dropped, NaN rows are replaced by mean'
        if 'df_y' in st.session_state:
            df_y = st.session_state['df_y']
            df_all = pd.concat([df_X.head(), df_y.head()], axis=1)
            caption = f'''
                {len(df_X)} samples, 
                {len(df_X.columns)} source features, 
                {len(df_y.columns)} target features
            '''
        st.caption(caption, help=help)
        st.dataframe(df_all, hide_index=True, height=100)
    else:
        st.write(
            '''
            To begin select you data source: 
            
            * Select an example to see how this works 
            * You can submit a csv to try on you data
            * Or you can use a publicly available dataset from 
              [OpenML](https://www.openml.org/search?type=data&sort=runs&status=active).
            ''')


def main():
    st.set_page_config(layout='wide', page_icon=MAPPER_EMOJI, page_title='tda-mapper-app', menu_items={
        'Report a bug': 'https://github.com/lucasimi/tda-mapper-python/issues',
        'About': 'https://github.com/lucasimi/tda-mapper-python/README.md'
    })
    st.title('tda-mapper-app')
    with st.sidebar:
        st.write('## 📊 Data Source')
        get_data()
        st.write(f'## {MAPPER_EMOJI} Mapper Settings')
        run_mapper()
    display_data_source()
    if 'mapper_graph' in st.session_state:
        render_graph()
    else:
        st.write(
            '''
            Experiment with Mapper Settings and hit Run when you're ready!
            ''')


main()
