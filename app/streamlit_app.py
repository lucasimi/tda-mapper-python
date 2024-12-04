import json
import time
import io
import gzip
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import networkx as nx
from networkx.readwrite.json_graph import adjacency_data

from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    KMeans,
    AffinityPropagation,
)
from sklearn.decomposition import PCA

from umap import UMAP

from tdamapper.core import ATTR_SIZE
from tdamapper.learn import MapperAlgorithm
from tdamapper.cover import CubicalCover, BallCover
from tdamapper.plot import MapperPlot
from tdamapper.utils.metrics import minkowski


OPENML_URL = 'https://www.openml.org/search?type=data&sort=runs&status=active'

S_RESULTS = 'stored_results'

MAX_SAMPLES = 1000

SAMPLE_FRAC = 0.1

V_DATA_SUMMARY_FEAT = 'feature'

V_DATA_SUMMARY_HIST = 'histogram'

V_DATA_SUMMARY_BINS = 15

V_LENS_IDENTITY = 'Identity'

V_LENS_PCA = 'PCA'

V_LENS_UMAP = 'UMAP'

V_COVER_TRIVIAL = 'Trivial'

V_COVER_BALL = 'Ball'

V_COVER_CUBICAL = 'Cubical'

V_CLUSTERING_TRIVIAL = 'Trivial'

V_CLUSTERING_AGGLOMERATIVE = 'Agglomerative'

V_CLUSTERING_DBSCAN = 'DBSCAN'

V_CLUSTERING_HDBSCAN = 'HDDBSCAN'

V_CLUSTERING_KMEANS = 'KMEANS'

V_CLUSTERING_AFFINITY_PROPAGATION = 'KMEANS'

VD_SEED = 42

V_AGGREGATION_MEAN = 'Mean'

V_AGGREGATION_STD = 'Std'

V_AGGREGATION_MODE = 'Mode'

V_AGGREGATION_QUANTILE = 'Quantile'

V_CMAP_JET = 'Jet (Sequential)'

V_CMAP_VIRIDIS = 'Viridis (Sequential)'

V_CMAP_CIVIDIS = 'Cividis (Sequential)'

V_CMAP_SPECTRAL = 'Spectral (Diverging)'

V_CMAP_PORTLAND = 'Portland (Diverging)'

V_CMAP_HSV = 'HSV (Cyclic)'

V_CMAP_TWILIGHT = 'Twilight (Cyclic)'

GIT_REPO_URL = 'https://github.com/lucasimi/tda-mapper-python'

ICON_URL = f'{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-icon.png'

LOGO_URL = f'{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png'

APP_TITLE = 'tda-mapper'

REPORT_BUG = f'{GIT_REPO_URL}/issues'

ABOUT = f'{GIT_REPO_URL}/blob/main/README.md'

FOOTER = (
    'If you find this app useful, please consider leaving a '
    f':star: on **[GitHub]({GIT_REPO_URL})**.'
)


def mode(arr):
    unique, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    return unique[max_count_index]


def quantile(q):
    return lambda agg: np.nanquantile(agg, q=q)


@st.cache_data
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


def _get_data_summary(df_X, df_y):
    df = pd.concat([get_sample(df_y), get_sample(df_X)], axis=1)
    df_hist = pd.DataFrame({
        x: df[x].value_counts(bins=V_DATA_SUMMARY_BINS, sort=False).values
        for x in df.columns
    }).T
    df_summary = pd.DataFrame({
        V_DATA_SUMMARY_FEAT: df.columns,
        V_DATA_SUMMARY_HIST: df_hist.values.tolist()})
    return df_summary


def initialize():
    st.set_page_config(
        layout='wide',
        page_icon=ICON_URL,
        page_title=APP_TITLE,
        menu_items={
            'Report a bug': REPORT_BUG,
            'About': ABOUT,
        },
    )
    st.logo(LOGO_URL, size='large', link=GIT_REPO_URL)


@st.cache_data(max_entries=1)
def load_data(source=None, name=None, csv=None):
    X, y = pd.DataFrame(), pd.DataFrame()
    if source == 'Example':
        if name == 'Digits':
            X, y = load_digits(return_X_y=True, as_frame=True)
        elif name == 'Iris':
            X, y = load_iris(return_X_y=True, as_frame=True)
    elif source == 'OpenML':
        X, y = fetch_openml(name, return_X_y=True, as_frame=True, parser='auto')
    elif source == 'CSV':
        if csv is not None:
            X, y = pd.read_csv(csv), pd.DataFrame()
    df_X, df_y = fix_data(X), fix_data(y)
    return df_X, df_y


def data_input_section():
    source = None
    name = None
    csv = None
    source = st.selectbox(
        '📦 Data Source',
        options=['Example', 'OpenML', 'CSV'],
    )
    if source == 'Example':
        name = st.selectbox('📦 Name', options=['Digits', 'Iris'])
    elif source == 'OpenML':
        name = st.text_input(
            '📦 Name',
            placeholder='Name',
            help=f'Search on [OpenML]({OPENML_URL})',
        )
    elif source == 'CSV':
        csv = st.file_uploader('Upload')
    return load_data(source, name, csv)


def mapper_lens_input_section(X):
    st.subheader('🔎 Lens')
    lens_type = st.selectbox(
        'Type',
        options=[
            V_LENS_IDENTITY,
            V_LENS_PCA,
            V_LENS_UMAP,
        ],
        index=1,
    )
    lens = None
    if lens_type == V_LENS_IDENTITY:
        lens = X
    elif lens_type == V_LENS_PCA:
        pca_n = st.number_input(
            'PCA Components',
            value=2,
            min_value=1,
        )
        _, n_feats = X.shape
        if pca_n > n_feats:
            lens = X
        else:
            lens = PCA(n_components=pca_n).fit_transform(X)
    elif lens_type == V_LENS_UMAP:
        umap_n = st.number_input(
            'UMAP Components',
            value=2,
            min_value=1,
        )
        _, n_feats = X.shape
        if umap_n > n_feats:
            lens = X
        else:
            lens = UMAP(n_components=umap_n).fit_transform(X)
    return lens


def mapper_cover_input_section():
    st.subheader('🌐 Cover')
    cover_type = st.selectbox(
        'Type',
        options=[V_COVER_TRIVIAL, V_COVER_BALL, V_COVER_CUBICAL],
        index=2,
    )
    cover = None
    if cover_type == V_COVER_TRIVIAL:
        cover = None
    elif cover_type == V_COVER_BALL:
        ball_r = st.number_input(
            'Radius',
            value=100.0,
            min_value=0.0,
        )
        ball_metric_p = st.number_input(
            '$L_p$ metric',
            value=2,
            min_value=1,
        )
        cover = BallCover(radius=ball_r, metric=minkowski(ball_metric_p))
    elif cover_type == V_COVER_CUBICAL:
        cubical_n = st.number_input(
            'Intervals',
            value=10,
            min_value=0)
        cubical_p = st.number_input(
            'Overlap',
            value=0.25,
            min_value=0.0,
            max_value=1.0)
        cover = CubicalCover(n_intervals=cubical_n, overlap_frac=cubical_p)
    return cover


def mapper_clustering_input_section():
    st.subheader('🧮 Clustering')
    clustering_type = st.selectbox(
        'Type',
        options=[V_CLUSTERING_TRIVIAL, V_CLUSTERING_AGGLOMERATIVE],
        index=1,
    )
    clustering = None
    if clustering_type == V_CLUSTERING_TRIVIAL:
        clustering = None
    elif clustering_type == V_CLUSTERING_AGGLOMERATIVE:
        clust_num = st.number_input(
            'Clusters',
            value=2,
            min_value=1,
        )
        linkage = st.selectbox(
            'Linkage',
            options=[
                'ward',
                'complete',
                'average',
                'single',
            ],
        )
        n_clusters = int(clust_num)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
    elif clustering_type == V_CLUSTERING_KMEANS:
        clust_num = st.number_input(
            'Clusters',
            value=2,
            min_value=1,
        )
        clustering = KMeans(n_clusters=n_clusters, n_init='auto')

    return clustering


def mapper_input_section(X):
    lens = mapper_lens_input_section(X)
    st.divider()
    cover = mapper_cover_input_section()
    st.divider()
    clustering = mapper_clustering_input_section()
    mapper_algo = MapperAlgorithm(
        cover=cover,
        clustering=clustering,
        verbose=False,
        n_jobs=-1,
    )
    mapper_graph = mapper_algo.fit_transform(X, lens)
    return mapper_graph


def plot_agg_input_section():
    agg_type = st.selectbox(
        'Aggregation',
        options=[
            V_AGGREGATION_MEAN,
            V_AGGREGATION_STD,
            V_AGGREGATION_MODE,
            V_AGGREGATION_QUANTILE,
        ],
    )
    agg = None
    agg_name = None
    if agg_type == V_AGGREGATION_MEAN:
        agg = np.nanmean
        agg_name = 'Mean'
    elif agg_type == V_AGGREGATION_STD:
        agg = np.nanstd
        agg_name = 'Std'
    elif agg_type == V_AGGREGATION_MODE:
        agg = mode
        agg_name = 'Mode'
    elif agg_type == V_AGGREGATION_QUANTILE:
        q = st.slider(
            'Rank',
            value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        agg = quantile(q)
        agg_name = f'Quantile {round(q, 2)}'
    return agg, agg_name


def plot_cmap_input_section():
    cmap_type = st.selectbox(
        'Colormap',
        options=[
            V_CMAP_JET,
            V_CMAP_VIRIDIS,
            V_CMAP_CIVIDIS,
            V_CMAP_PORTLAND,
            V_CMAP_SPECTRAL,
            V_CMAP_HSV,
            V_CMAP_TWILIGHT,
        ],
    )
    cmap = None
    if cmap_type == V_CMAP_JET:
        cmap = 'Jet'
    elif cmap_type == V_CMAP_VIRIDIS:
        cmap = 'Viridis'
    elif cmap_type == V_CMAP_CIVIDIS:
        cmap = 'Cividis'
    elif cmap_type == V_CMAP_PORTLAND:
        cmap = 'Portland'
    elif cmap_type == V_CMAP_SPECTRAL:
        cmap = 'Spectral'
    elif cmap_type == V_CMAP_HSV:
        cmap = 'HSV'
    elif cmap_type == V_CMAP_TWILIGHT:
        cmap = 'Twilight'
    return cmap


def plot_color_input_section(df_X, df_y):
    X_cols = list(df_X.columns)
    y_cols = list(df_y.columns)
    col_feat = st.selectbox(
        'Color',
        options=y_cols + X_cols,
    )
    if col_feat in X_cols:
        return df_X[col_feat].to_numpy(), col_feat
    elif col_feat in y_cols:
        return df_y[col_feat].to_numpy(), col_feat


def plot_seed_input_section():
    seed = st.number_input(
        'Seed',
        value=VD_SEED,
        help='Changing this value alters the shape',
    )
    return seed


def plot_dim_input_section():
    toggle_3d = st.toggle(
        '3D',
        value=True,
    )
    dim = 3 if toggle_3d else 2
    return dim


def plot_input_section(df_X, df_y, mapper_graph):
    st.subheader('🎨 Drawing')
    dim = plot_dim_input_section()
    seed = plot_seed_input_section()
    agg, agg_name = plot_agg_input_section()
    cmap = plot_cmap_input_section()
    colors, colors_feat = plot_color_input_section(df_X, df_y)
    mapper_plot = MapperPlot(mapper_graph, dim, seed=seed)
    mapper_fig = mapper_plot.plot_plotly(
        colors,
        agg=agg,
        title=f'{agg_name} of {colors_feat}',
        cmap=cmap,
        width=600,
        height=600,
    )
    mapper_fig.update_layout(
        dragmode='pan' if dim == 2 else 'orbit',
        uirevision='constant',
        margin=dict(b=0, l=0, r=0, t=0),
    )
    mapper_fig.update_xaxes(
        showline=False,
    )
    mapper_fig.update_yaxes(
        showline=False,
        scaleanchor='x',
        scaleratio=1,
    )
    return mapper_plot, mapper_fig


def mapper_rendering_section(mapper_fig):
    config = {'scrollZoom': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['zoom', 'pan']}
    st.plotly_chart(mapper_fig, use_container_width=True, config=config)


def data_summary_section(df_X, df_y):
    df_summary = _get_data_summary(df_X, df_y)
    st.dataframe(
        df_summary,
        hide_index=True,
        height=600,
        column_config={
            V_DATA_SUMMARY_HIST: st.column_config.AreaChartColumn(
                width='large',
            ),
            V_DATA_SUMMARY_FEAT: st.column_config.TextColumn(
                width='small',
                disabled=True,
            )
        },
        use_container_width=True,
    )


def get_gzip_bytes(string, encoding='utf-8'):
    fileobj = io.BytesIO()
    gzf = gzip.GzipFile(fileobj=fileobj, mode='wb', compresslevel=6)
    gzf.write(string.encode(encoding))
    gzf.close()
    return fileobj.getvalue()


def data_download_button(df_X, df_y):
    df_all = pd.concat([df_X, df_y])
    buffer = io.BytesIO()
    df_all.to_csv(buffer, index=False, compression={'method': 'gzip', 'compresslevel': 6})
    buffer.seek(0)
    timestamp = int(time.time())
    human_readable = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    st.download_button(
        '📥 Download Data',
        disabled=df_all.empty,
        use_container_width=True,
        data=buffer,
        file_name=f'data_{human_readable}.gzip',
        mime='text/csv',
    )


def mapper_download_button(mapper_graph):
    mapper_adj = {} if mapper_graph is None else adjacency_data(mapper_graph)
    buffer = io.BytesIO()
    mapper_json = json.dumps(mapper_adj, default=int)
    buffer.write(mapper_json.encode('utf-8'))
    buffer.seek(0)
    timestamp = int(time.time())
    human_readable = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
    download_button = st.download_button(
        '📥 Download Mapper Graph',
        data=get_gzip_bytes(mapper_json),
        disabled=mapper_graph is None,
        use_container_width=True,
        file_name=f'mapper_graph_{human_readable}.gzip',
    )
    return download_button


def main():
    initialize()
    with st.sidebar:
        try:
            df_X, df_y = data_input_section()
            st.toast('Successfully Loaded Data', icon='📦')
        except ValueError as err:
            st.toast(f'# {err}', icon='🚨')
            df_X, df_y = pd.DataFrame(), pd.DataFrame() 
        st.divider()
        mapper_graph = mapper_input_section(df_X.to_numpy())
        st.divider()
        mapper_plot, mapper_fig = plot_input_section(df_X, df_y, mapper_graph)
    col_0, col_1 = st.columns([1, 3])
    with col_0:
        data_summary_section(df_X, df_y)
    with col_1:
        mapper_rendering_section(mapper_fig)
    col_2, col_3 = st.columns([1, 3])
    with col_2:
        data_download_button(df_X, df_y)
    with col_3:
        mapper_download_button(mapper_graph)
    st.divider()
    st.markdown(FOOTER)


main()
