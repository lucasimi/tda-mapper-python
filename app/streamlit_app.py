import gzip
import io
import json
import logging
import os
import time
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from networkx.readwrite.json_graph import adjacency_data
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
)
from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.decomposition import PCA
from umap import UMAP

from tdamapper._plot_plotly import _marker_size
from tdamapper.core import aggregate_graph
from tdamapper.cover import BallCover, CubicalCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

LIMITS_ENABLED = bool(os.environ.get("LIMITS_ENABLED", False))

LIMITS_NUM_SAMPLES = int(os.environ.get("LIMITS_NUM_SAMPLES", 10000))

LIMITS_NUM_FEATURES = int(os.environ.get("LIMITS_NUM_FEATURES", 1000))

LIMITS_NUM_NODES = int(os.environ.get("LIMITS_NUM_NODES", 2000))

LIMITS_NUM_EDGES = int(os.environ.get("LIMITS_NUM_EDGES", 3000))

OPENML_URL = "https://www.openml.org/search?type=data&sort=runs&status=active"

S_RESULTS = "stored_results"

MAX_SAMPLES = 1000

SAMPLE_FRAC = 0.1

V_DATA_SUMMARY_FEAT = "Feat"

V_DATA_SUMMARY_HIST = "Hist"

V_DATA_SUMMARY_BINS = 15

V_LENS_IDENTITY = "Identity"

V_LENS_PCA = "PCA"

V_LENS_UMAP = "UMAP"

V_COVER_TRIVIAL = "Trivial"

V_COVER_BALL = "Ball"

V_COVER_CUBICAL = "Cubical"

V_CLUSTERING_TRIVIAL = "Trivial"

V_CLUSTERING_AGGLOMERATIVE = "Agglomerative"

V_CLUSTERING_DBSCAN = "DBSCAN"

V_CLUSTERING_HDBSCAN = "HDBSCAN"

V_CLUSTERING_KMEANS = "KMeans"

V_CLUSTERING_AFFINITY_PROPAGATION = "Affinity Propagation"

VD_SEED = 42

VD_ITERATIONS = 50

V_AGGREGATION_MEAN = "Mean"

V_AGGREGATION_STD = "Std"

V_AGGREGATION_MODE = "Mode"

V_AGGREGATION_QUANTILE = "Quantile"

V_CMAP_JET = "Jet (Sequential)"

V_CMAP_VIRIDIS = "Viridis (Sequential)"

V_CMAP_CIVIDIS = "Cividis (Sequential)"

V_CMAP_SPECTRAL = "Spectral (Diverging)"

V_CMAP_PORTLAND = "Portland (Diverging)"

V_CMAP_HSV = "HSV (Cyclic)"

V_CMAP_TWILIGHT = "Twilight (Cyclic)"

V_CMAPS = {
    V_CMAP_JET: "Jet",
    V_CMAP_VIRIDIS: "Viridis",
    V_CMAP_CIVIDIS: "Cividis",
    V_CMAP_SPECTRAL: "Spectral",
    V_CMAP_PORTLAND: "Portland",
    V_CMAP_HSV: "HSV",
    V_CMAP_TWILIGHT: "Twilight",
}

GIT_REPO_URL = "https://github.com/lucasimi/tda-mapper-python"

ICON_URL = f"{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-icon.png"

LOGO_URL = f"{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png"

APP_TITLE = "tda-mapper"

REPORT_BUG = f"{GIT_REPO_URL}/issues"

ABOUT = f"{GIT_REPO_URL}/blob/main/README.md"

FOOTER = (
    "If you find this app useful, please consider leaving a "
    f":star: on **[GitHub]({GIT_REPO_URL})**."
)


logger = st.logger.get_logger(__name__)


def _check_limits_mapper_graph(mapper_graph):
    if LIMITS_ENABLED:
        num_nodes = mapper_graph.number_of_nodes()
        if num_nodes > LIMITS_NUM_NODES:
            logging.warn("Too many nodes.")
            raise ValueError(
                "Too many nodes: select different parameters or run the app "
                "locally on your machine."
            )
        num_edges = mapper_graph.number_of_edges()
        if num_edges > LIMITS_NUM_EDGES:
            logging.warn("Too many edges.")
            raise ValueError(
                "Too many edges: select different parameters or run the app "
                "locally on your machine."
            )


def _check_limits_dataset(df_X, df_y):
    if LIMITS_ENABLED:
        num_samples = len(df_X)
        if num_samples > LIMITS_NUM_SAMPLES:
            logging.warn("Dataset too big.")
            raise ValueError(
                "Dataset too big: select a different dataset or run the app "
                "locally on your machine."
            )
        num_features = len(df_X.columns) + len(df_y.columns)
        if num_features > LIMITS_NUM_FEATURES:
            logging.warn("Too many features.")
            raise ValueError(
                "Too many features: select a different dataset or run the app "
                "locally on your machine."
            )


def _fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include="number")
    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def _get_graph_no_attribs(graph):
    graph_no_attribs = nx.Graph()
    graph_no_attribs.add_nodes_from(graph.nodes())
    graph_no_attribs.add_edges_from(graph.edges())
    return graph_no_attribs


def _encode_graph(graph):
    nodes = tuple(sorted([int(v) for v in graph.nodes()]))
    edges = tuple(sorted(tuple(sorted(e)) for e in graph.edges()))
    return (nodes, edges)


def _get_data_summary(df_X, df_y):
    df = pd.concat([get_sample(df_y), get_sample(df_X)], axis=1)
    df_hist = pd.DataFrame(
        {
            x: df[x].value_counts(bins=V_DATA_SUMMARY_BINS, sort=False).values
            for x in df.columns
        }
    ).T
    df_summary = pd.DataFrame(
        {V_DATA_SUMMARY_FEAT: df.columns, V_DATA_SUMMARY_HIST: df_hist.values.tolist()}
    )
    return df_summary


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


def initialize():
    st.set_page_config(
        layout="wide",
        page_icon=ICON_URL,
        page_title=APP_TITLE,
        menu_items={
            "Report a bug": REPORT_BUG,
            "About": ABOUT,
        },
    )
    st.logo(LOGO_URL, size="large", link=GIT_REPO_URL)
    with st.sidebar:
        st.markdown("*Explore data with Mapper*")
        st.header("")


@st.cache_data(max_entries=1)
def load_data(source=None, name=None, csv=None):
    X, y = pd.DataFrame(), pd.DataFrame()
    if source == "Example":
        if name == "Digits":
            X, y = load_digits(return_X_y=True, as_frame=True)
        elif name == "Iris":
            X, y = load_iris(return_X_y=True, as_frame=True)
    elif source == "OpenML":
        logging.info(f"Fetching dataset {name} from OpenML")
        X, y = fetch_openml(
            name,
            return_X_y=True,
            as_frame=True,
            parser="auto",
        )
    elif source == "CSV":
        if csv is None:
            raise ValueError("No csv file uploaded")
        else:
            X, y = pd.read_csv(csv), pd.DataFrame()
    df_X, df_y = _fix_data(X), _fix_data(y)
    _check_limits_dataset(df_X, df_y)
    return df_X, df_y


def data_input_section():
    source = None
    name = None
    csv = None
    st.header("📊 Data")
    source = st.selectbox(
        "Source",
        options=["Example", "OpenML", "CSV"],
    )
    if source == "Example":
        name = st.selectbox("Name", options=["Digits", "Iris"])
    elif source == "OpenML":
        name = st.text_input(
            "Name",
            placeholder="Name",
            help=f"Search on [OpenML]({OPENML_URL})",
        )
    elif source == "CSV":
        csv = st.file_uploader("Upload")
    return load_data(source, name, csv)


def mapper_lens_input_section(X):
    st.header("🔎 Lens")
    lens_type = st.selectbox(
        "Type",
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
            "PCA Components",
            value=2,
            min_value=1,
        )
        pca_random_state = st.number_input(
            "PCA random state",
            value=VD_SEED,
        )
        _, n_feats = X.shape
        if pca_n > n_feats:
            lens = X
        else:
            lens = PCA(n_components=pca_n, random_state=pca_random_state).fit_transform(
                X
            )
    elif lens_type == V_LENS_UMAP:
        umap_n = st.number_input(
            "UMAP Components",
            value=2,
            min_value=1,
        )
        umap_random_state = st.number_input(
            "UMAP random state",
            value=VD_SEED,
        )
        _, n_feats = X.shape
        if umap_n > n_feats:
            lens = X
        else:
            lens = UMAP(
                n_components=umap_n, random_state=umap_random_state
            ).fit_transform(X)
    return lens


def mapper_cover_input_section():
    st.header("🌐 Cover")
    cover_type = st.selectbox(
        "Type",
        options=[V_COVER_TRIVIAL, V_COVER_BALL, V_COVER_CUBICAL],
        index=2,
    )
    cover = None
    if cover_type == V_COVER_TRIVIAL:
        cover = None
    elif cover_type == V_COVER_BALL:
        ball_r = st.number_input(
            "Radius",
            value=100.0,
            min_value=0.0,
        )
        metric = st.selectbox(
            "Metric",
            options=[
                "euclidean",
                "chebyshev",
                "manhattan",
                "cosine",
            ],
            key="cover_metric",
        )
        cover = BallCover(radius=ball_r, metric=metric)
    elif cover_type == V_COVER_CUBICAL:
        cubical_n = st.number_input("Intervals", value=10, min_value=0)
        cubical_overlap = st.checkbox(
            "Set overlap",
            value=False,
            help="Uses a dimension-dependant default overlap when unchecked",
        )
        cubical_p = None
        if cubical_overlap:
            cubical_p = st.number_input(
                "Overlap", value=0.25, min_value=0.0, max_value=1.0
            )
        cover = CubicalCover(n_intervals=cubical_n, overlap_frac=cubical_p)
    return cover


def mapper_clustering_kmeans():
    clust_num = st.number_input(
        "Clusters",
        value=2,
        min_value=1,
    )
    n_clusters = int(clust_num)
    return KMeans(n_clusters=n_clusters, n_init="auto")


def mapper_clustering_dbscan():
    eps = st.number_input(
        "Eps",
        value=0.5,
        min_value=0.0,
    )
    min_samples = st.number_input(
        "Min Samples",
        value=5,
        min_value=1,
    )
    metric = st.selectbox(
        "Metric",
        options=[
            "euclidean",
            "chebyshev",
            "manhattan",
            "cosine",
        ],
    )
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric)


def mapper_clustering_hdbscan():
    min_cluster_size = st.number_input(
        "Min Cluster Size",
        value=5,
        min_value=1,
    )
    metric = st.selectbox(
        "Metric",
        options=[
            "euclidean",
            "chebyshev",
            "manhattan",
        ],
    )
    return HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)


def mapper_clustering_agglomerative():
    clust_num = st.number_input(
        "Clusters",
        value=2,
        min_value=1,
    )
    linkage = st.selectbox(
        "Linkage",
        options=[
            "ward",
            "complete",
            "average",
            "single",
        ],
        index=3,
    )
    n_clusters = int(clust_num)
    metric = st.selectbox(
        "Metric",
        options=[
            "euclidean",
            "chebyshev",
            "manhattan",
            "cosine",
        ],
    )
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
    )


def mapper_clustering_affinityprop():
    damping = st.number_input(
        "Damping",
        value=0.5,
        min_value=0.0,
    )
    max_iter = st.number_input(
        "Max Iter",
        value=200,
        min_value=50,
    )
    return AffinityPropagation(damping=damping, max_iter=max_iter)


def mapper_clustering_input_section():
    st.header("🧮 Clustering")
    clustering_type = st.selectbox(
        "Type",
        options=[
            V_CLUSTERING_TRIVIAL,
            V_CLUSTERING_KMEANS,
            V_CLUSTERING_AGGLOMERATIVE,
            V_CLUSTERING_DBSCAN,
            V_CLUSTERING_HDBSCAN,
            V_CLUSTERING_AFFINITY_PROPAGATION,
        ],
        index=1,
    )
    clustering = None
    if clustering_type == V_CLUSTERING_TRIVIAL:
        clustering = None
    elif clustering_type == V_CLUSTERING_AGGLOMERATIVE:
        clustering = mapper_clustering_agglomerative()
    elif clustering_type == V_CLUSTERING_KMEANS:
        clustering = mapper_clustering_kmeans()
    elif clustering_type == V_CLUSTERING_DBSCAN:
        clustering = mapper_clustering_dbscan()
    elif clustering_type == V_CLUSTERING_HDBSCAN:
        clustering = mapper_clustering_hdbscan()
    elif clustering_type == V_CLUSTERING_AFFINITY_PROPAGATION:
        clustering = mapper_clustering_affinityprop()
    return clustering


@st.cache_data(
    hash_funcs={"tdamapper.learn.MapperAlgorithm": MapperAlgorithm.__repr__},
    show_spinner="Computing Mapper",
)
def compute_mapper(mapper, X, y):
    logger.info("Generating Mapper graph")
    mapper_graph = mapper.fit_transform(X, y)
    return mapper_graph


def mapper_input_section(X):
    lens = mapper_lens_input_section(X)
    st.divider()
    cover = mapper_cover_input_section()
    st.divider()
    clustering = mapper_clustering_input_section()
    mapper_algo = MapperAlgorithm(
        cover=cover,
        clustering=clustering,
        verbose=True,
        n_jobs=1,
    )
    mapper_graph = compute_mapper(mapper_algo, X, lens)
    return mapper_graph


def plot_agg_input_section():
    agg_type = st.selectbox(
        "Aggregation",
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
        agg_name = "Mean"
    elif agg_type == V_AGGREGATION_STD:
        agg = np.nanstd
        agg_name = "Std"
    elif agg_type == V_AGGREGATION_MODE:
        agg = mode
        agg_name = "Mode"
    elif agg_type == V_AGGREGATION_QUANTILE:
        q = st.slider(
            "Rank",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
        )
        agg = quantile(q)
        agg_name = f"Quantile {round(q, 2)}"
    return agg, agg_name


@st.cache_data(
    hash_funcs={
        "networkx.classes.graph.Graph": lambda g: _encode_graph(
            _get_graph_no_attribs(g)
        )
    },
    show_spinner="Generating Mapper Layout",
)
def compute_mapper_plot(mapper_graph, dim, seed, iterations):
    _check_limits_mapper_graph(mapper_graph)
    logger.info("Generating Mapper plot")
    mapper_plot = MapperPlot(
        mapper_graph,
        dim,
        seed=seed,
        iterations=iterations,
    )
    return mapper_plot


def mapper_plot_section(mapper_graph):
    st.header("🗺️ Layout")
    toggle_3d = st.toggle(
        "3D",
        value=True,
    )
    dim = 3 if toggle_3d else 2
    seed = st.number_input(
        "Seed",
        value=VD_SEED,
    )
    iterations = st.number_input(
        "Iterations",
        value=VD_ITERATIONS,
        min_value=1,
    )
    mapper_plot = compute_mapper_plot(
        mapper_graph,
        dim,
        seed=seed,
        iterations=iterations,
    )
    return mapper_plot


@st.cache_data(
    hash_funcs={"tdamapper.plot.MapperPlot": lambda mp: mp.positions},
    show_spinner="Rendering Mapper",
)
def compute_mapper_fig(mapper_plot, colors, node_size, cmap, _agg, agg_name):
    logger.info("Generating Mapper figure")
    mapper_fig = mapper_plot.plot_plotly(
        colors,
        color_names=colors.columns,
        node_size=node_size,
        agg=_agg,
        title=agg_name,
        cmap=cmap,
        width=600,
        height=600,
    )
    return mapper_fig


def mapper_figure_section(df_X, df_y, mapper_plot):
    st.header("🎨 Plot")
    agg, agg_name = plot_agg_input_section()
    colors = pd.concat([df_y, df_X], axis=1)
    mapper_fig = compute_mapper_fig(
        mapper_plot,
        colors=colors,
        node_size=1.0,
        _agg=agg,
        cmap=["Jet", "Viridis", "Cividis"],
        agg_name=agg_name,
    )
    mapper_fig.update_layout(
        margin=dict(b=5, l=5, r=5, t=5),
    )
    mapper_fig.update_xaxes(
        showline=False,
    )
    mapper_fig.update_yaxes(
        showline=False,
        scaleanchor="x",
        scaleratio=1,
    )

    return mapper_fig


def _compute_colors_agg(mapper_plot, df_X, df_y, col_feat, agg):
    X_cols = list(df_X.columns)
    y_cols = list(df_y.columns)
    colors = np.array([])
    if col_feat in X_cols:
        colors = df_X[col_feat].to_numpy()
    elif col_feat in y_cols:
        colors = df_y[col_feat].to_numpy()
    return aggregate_graph(colors, mapper_plot.graph, agg)


def _edge_colors(mapper_plot, df_X, df_y, col_feat, agg):
    colors_avg = []
    colors_agg = _compute_colors_agg(mapper_plot, df_X, df_y, col_feat, agg)
    for edge in mapper_plot.graph.edges():
        c0, c1 = colors_agg[edge[0]], colors_agg[edge[1]]
        colors_avg.append(c0)
        colors_avg.append(c1)
        colors_avg.append(c1)
    return colors_avg


def mapper_rendering_section(mapper_fig):
    config = {
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["zoom", "pan"],
    }
    st.plotly_chart(
        mapper_fig, use_container_width=True, config=config, key="mapper_plot"
    )


def data_summary_section(df_X, df_y, mapper_graph):
    df_stats = pd.DataFrame(
        {
            "Stat": [
                "Samples",
                "Input Feats",
                "Target Feats",
                "Nodes",
                "Edges",
                "Conn. Comp.",
            ],
            "Value": [
                len(df_X),
                len(df_X.columns),
                len(df_y.columns),
                mapper_graph.number_of_nodes(),
                mapper_graph.number_of_edges(),
                nx.number_connected_components(mapper_graph),
            ],
        }
    )
    st.dataframe(
        df_stats,
        hide_index=True,
        use_container_width=True,
        height=250,
    )
    df_summary = _get_data_summary(df_X, df_y)
    st.dataframe(
        df_summary,
        hide_index=True,
        height=330,
        column_config={
            V_DATA_SUMMARY_HIST: st.column_config.AreaChartColumn(
                width="small",
            ),
            V_DATA_SUMMARY_FEAT: st.column_config.TextColumn(
                width="small",
                disabled=True,
            ),
        },
        use_container_width=True,
    )


def get_gzip_bytes(string, encoding="utf-8"):
    fileobj = io.BytesIO()
    gzf = gzip.GzipFile(fileobj=fileobj, mode="wb", compresslevel=6)
    gzf.write(string.encode(encoding))
    gzf.close()
    return fileobj.getvalue()


def data_download_button(df_X, df_y):
    df_all = pd.concat([df_X, df_y])
    buffer = io.BytesIO()
    df_all.to_csv(
        buffer,
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
        },
    )
    buffer.seek(0)
    timestamp = int(time.time())
    dt = datetime.fromtimestamp(timestamp)
    human_readable = dt.strftime("%Y-%m-%d_%H-%M-%S")
    st.download_button(
        "📥 Download Data",
        disabled=df_all.empty,
        use_container_width=True,
        data=buffer,
        file_name=f"data_{human_readable}.gzip",
        mime="text/csv",
    )


def mapper_download_button(mapper_graph):
    mapper_adj = {} if mapper_graph is None else adjacency_data(mapper_graph)
    buffer = io.BytesIO()
    mapper_json = json.dumps(mapper_adj, default=int)
    buffer.write(mapper_json.encode("utf-8"))
    buffer.seek(0)
    timestamp = int(time.time())
    dt = datetime.fromtimestamp(timestamp)
    human_readable = dt.strftime("%Y-%m-%d_%H-%M-%S")
    download_button = st.download_button(
        "📥 Download Mapper Graph",
        data=get_gzip_bytes(mapper_json),
        disabled=mapper_graph is None,
        use_container_width=True,
        file_name=f"mapper_graph_{human_readable}.gzip",
    )
    return download_button


def main():
    initialize()
    with st.sidebar:
        try:
            df_X, df_y = data_input_section()
        except ValueError as err:
            st.error(f"{err}")
            return
        st.divider()
        mapper_graph = mapper_input_section(df_X.to_numpy())
        st.divider()
        mapper_plot = mapper_plot_section(mapper_graph)
        st.divider()
        mapper_fig = mapper_figure_section(df_X, df_y, mapper_plot)
    col_0, col_1 = st.columns([1, 3])
    with col_0:
        data_summary_section(df_X, df_y, mapper_graph)
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
