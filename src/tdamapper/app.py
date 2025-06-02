import logging
from dataclasses import asdict, dataclass

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from nicegui import run, ui
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import fetch_openml, load_digits, load_iris
from sklearn.decomposition import PCA
from umap import UMAP

from tdamapper.core import TrivialClustering, TrivialCover
from tdamapper.cover import BallCover, CubicalCover, KNNCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

LENS_IDENTITY = "Identity"
LENS_PCA = "PCA"
LENS_UMAP = "UMAP"

COVER_TRIVIAL = "Trivial"
COVER_CUBICAL = "Cubical"
COVER_BALL = "Ball"
COVER_KNN = "KNN"

CLUSTERING_TRIVIAL = "Trivial"
CLUSTERING_KMEANS = "KMeans"
CLUSTERING_AGGLOMERATIVE = "Agglomerative"
CLUSTERING_DBSCAN = "DBSCAN"

DATA_SOURCE_EXAMPLE = "Example"
DATA_SOURCE_CSV = "CSV"
DATA_SOURCE_OPENML = "OpenML"

DATA_SOURCE_EXAMPLE_DIGITS = "Digits"
DATA_SOURCE_EXAMPLE_IRIS = "Iris"
SOURCE_OPENML = "554"

DRAW_3D = "3D"
DRAW_2D = "2D"
DRAW_ITERATIONS = 50
DRAW_MEAN = "Mean"
DRAW_MEDIAN = "Median"
DRAW_MODE = "Mode"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class State:
    source_type: str = DATA_SOURCE_EXAMPLE
    source_name: str = DATA_SOURCE_EXAMPLE_DIGITS
    source_csv: str = "tmp/data.csv"
    source_openml: str = SOURCE_OPENML
    lens_type: str = LENS_PCA
    lens_pca_n_components: int = 2
    lens_umap_n_components: int = 2
    cover_type: str = COVER_CUBICAL
    cover_cubical_n_intervals: int = 2
    cover_cubical_overlap_frac: float = 0.25
    cover_knn_neighbors: int = 10
    cover_ball_radius: float = 100.0
    clustering_type: str = CLUSTERING_TRIVIAL
    clustering_kmeans_n_clusters: int = 2
    clustering_dbscan_eps: float = 0.5
    clustering_dbscan_min_samples: int = 5
    clustering_agglomerative_n_clusters: int = 2
    draw_dim: int = 3
    draw_aggregation: str = DRAW_MEAN
    draw_iterations: int = 50


def _fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include="number")
    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def get_dataset(state: State):
    source_type = state.source_type
    source_name = state.source_name
    csv_file = state.source_csv
    openml_code = state.source_openml
    df_X, df_y = pd.DataFrame(), pd.Series()
    if source_type == DATA_SOURCE_EXAMPLE:
        if source_name == DATA_SOURCE_EXAMPLE_DIGITS:
            df_X, df_y = load_digits(return_X_y=True, as_frame=True)
        elif source_name == DATA_SOURCE_EXAMPLE_IRIS:
            df_X, df_y = load_iris(return_X_y=True, as_frame=True)
    elif source_type == DATA_SOURCE_CSV:
        if csv_file is None:
            logger.warning("No CSV file uploaded")
            df_X, df_y = pd.DataFrame(), pd.Series()
        else:
            df_X = pd.read_csv()
            df_y = pd.Series()
    elif source_type == DATA_SOURCE_OPENML:
        if not openml_code:
            logger.warning("No OpenML code provided")
            df_X, df_y = pd.DataFrame(), pd.Series()
        else:
            df_X, df_y = fetch_openml(openml_code, return_X_y=True, as_frame=True)
    else:
        logger.error(f"Unknown data source type: {source_type}")
        return pd.DataFrame(), pd.Series()
    df_X = _fix_data(df_X)
    df_y = _fix_data(df_y)
    return df_X, df_y


def get_lens(state: State):
    lens_type = state.lens_type
    if lens_type == LENS_IDENTITY:
        return lambda X: X
    elif lens_type == LENS_PCA:
        n_components = int(state.lens_pca_n_components)
        pca = PCA(n_components=n_components, random_state=42)
        return lambda X: pca.fit_transform(X)
    elif lens_type == LENS_UMAP:
        n_components = int(state.lens_umap_n_components)
        umap = UMAP(n_components=n_components, random_state=42)
        return lambda X: umap.fit_transform(X)
    else:
        logger.error(f"Unknown lens type: {lens_type}")
        return None


def get_cover(state: State):
    cover_type = state.cover_type
    if cover_type == COVER_TRIVIAL:
        return TrivialCover()
    elif cover_type == COVER_CUBICAL:
        n_intervals = int(state.cover_cubical_n_intervals)
        overlap_frac = float(state.cover_cubical_overlap_frac)
        return CubicalCover(n_intervals=n_intervals, overlap_frac=overlap_frac)
    elif cover_type == COVER_BALL:
        radius = float(state.cover_ball_radius)
        return BallCover(radius=radius)
    elif cover_type == COVER_KNN:
        neighbors = int(state.cover_knn_neighbors)
        return KNNCover(neighbors=neighbors)
    else:
        logger.error(f"Unknown cover type: {cover_type}")
        return None


def get_clustering(state: State):
    clustering_type = state.clustering_type
    if clustering_type == CLUSTERING_TRIVIAL:
        return TrivialClustering()
    elif clustering_type == CLUSTERING_KMEANS:
        n_clusters = state.clustering_kmeans_n_clusters
        return KMeans(n_clusters=n_clusters, random_state=42)
    elif clustering_type == CLUSTERING_DBSCAN:
        eps = state.clustering_dbscan_eps
        min_samples = state.clustering_dbscan_min_samples
        return DBSCAN(eps=eps, min_samples=min_samples)
    elif clustering_type == CLUSTERING_AGGLOMERATIVE:
        n_clusters = state.clustering_agglomerative_n_clusters
        return AgglomerativeClustering(n_clusters=n_clusters)
    else:
        logger.error(f"Unknown clustering type: {clustering_type}")
        return None


def compute_mapper(**kwargs):
    state = State(**kwargs)

    df_X, labels = get_dataset(state)
    if df_X.empty:
        logger.warning("No dataset loaded")
        return None

    lens = get_lens(state)
    if lens is None:
        logger.warning("No lens selected")
        return None

    X = df_X.to_numpy()
    y = lens(X)

    cover = get_cover(state)
    if cover is None:
        logger.warning("No cover selected")
        return None

    clustering = get_clustering(state)
    if clustering is None:
        logger.warning("No clustering selected")
        return None

    mapper_algo = MapperAlgorithm(
        cover=cover,
        clustering=clustering,
        verbose=False,
    )
    logger.info(f"Mapper configuration: {mapper_algo}")
    mapper_graph = mapper_algo.fit_transform(X, y)

    mapper_plot = MapperPlot(
        mapper_graph,
        dim=state.draw_dim,
        iterations=state.draw_iterations,
        seed=42,
    )

    colors = pd.concat([labels, df_X], axis=1)
    colors_arr = colors.to_numpy()
    color_names = colors.columns.tolist()

    mapper_fig = mapper_plot.plot_plotly(
        colors=colors_arr,
        cmap=["jet", "viridis", "cividis"],
        agg=np.nanmean,
        title=color_names,
        width=800,
        height=800,
        node_size=list(0.125 * x for x in range(17)),
    )
    mapper_fig.layout.width = None
    mapper_fig.layout.autosize = True

    return mapper_graph, mapper_fig


def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    mode_values = values[counts == max_count]
    return np.nanmean(mode_values)


def update_csv(file):
    if file is None:
        logger.warning("No file uploaded")
        return None
    return file


class App:

    def build_dataset(self):
        self.data_source_type = ui.select(
            label="Data Source",
            options=[
                DATA_SOURCE_EXAMPLE,
                DATA_SOURCE_CSV,
                DATA_SOURCE_OPENML,
            ],
            value=DATA_SOURCE_EXAMPLE,
            on_change=self.update,
        ).classes("w-full")
        self.data_source_example_file = ui.select(
            label="File",
            options=[
                DATA_SOURCE_EXAMPLE_DIGITS,
                DATA_SOURCE_EXAMPLE_IRIS,
            ],
            value=DATA_SOURCE_EXAMPLE_DIGITS,
            on_change=self.update,
        ).classes("w-full")
        self.data_source_example_file.bind_visibility_from(
            target_object=self.data_source_type,
            target_name="value",
            value=DATA_SOURCE_EXAMPLE,
        )
        self.data_source_csv = ui.upload(
            on_upload=self.update,
            auto_upload=True,
            label="Upload CSV",
        ).classes("w-full")
        self.data_source_csv.bind_visibility_from(
            target_object=self.data_source_type,
            target_name="value",
            value=DATA_SOURCE_CSV,
        )
        self.data_source_openml = ui.input(
            label="OpenML Code",
            on_change=self.update,
        ).classes("w-full")
        self.data_source_openml.bind_visibility_from(
            target_object=self.data_source_type,
            target_name="value",
            value=DATA_SOURCE_OPENML,
        )

    def build_lens(self):
        self.lens_type = ui.select(
            label="Lens type",
            options=[
                LENS_IDENTITY,
                LENS_PCA,
                LENS_UMAP,
            ],
            value=LENS_PCA,
            on_change=self.update,
        ).classes("w-full")
        self.pca_n_components = ui.number(
            label="PCA Components",
            min=1,
            max=10,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.pca_n_components.bind_visibility_from(
            target_object=self.lens_type,
            target_name="value",
            value=LENS_PCA,
        )
        self.umap_n_components = ui.number(
            label="UMAP Components",
            min=1,
            max=10,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.umap_n_components.bind_visibility_from(
            target_object=self.lens_type,
            target_name="value",
            value=LENS_UMAP,
        )

    def build_cover(self):
        self.cover_type = ui.select(
            label="Cover type",
            options=[
                COVER_TRIVIAL,
                COVER_CUBICAL,
                COVER_BALL,
                COVER_KNN,
            ],
            value=COVER_CUBICAL,
            on_change=self.update,
        ).classes("w-full")
        self.cover_cubical_n_intervals = ui.number(
            label="Intervals",
            min=1,
            max=100,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.cover_cubical_n_intervals.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_CUBICAL,
        )
        self.cover_cubical_overlap_frac = ui.number(
            label="Overlap",
            min=0.0,
            max=0.5,
            value=0.25,
            on_change=self.update,
        ).classes("w-full")
        self.cover_cubical_overlap_frac.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_CUBICAL,
        )
        self.cover_ball_radius = ui.number(
            label="Radius",
            min=0.0,
            value=100.0,
            on_change=self.update,
        ).classes("w-full")
        self.cover_ball_radius.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_BALL,
        )
        self.cover_knn_neighbors = ui.number(
            label="Neighbors",
            min=0,
            value=10,
            on_change=self.update,
        ).classes("w-full")
        self.cover_knn_neighbors.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_KNN,
        )

    def build_clustering(self):
        self.clustering_type = ui.select(
            label="Clustering type",
            options=[
                CLUSTERING_TRIVIAL,
                CLUSTERING_KMEANS,
                CLUSTERING_AGGLOMERATIVE,
                CLUSTERING_DBSCAN,
            ],
            value=CLUSTERING_TRIVIAL,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_kmeans_n_clusters = ui.number(
            label="Clusters",
            min=1,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_kmeans_n_clusters.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_KMEANS,
        )
        self.clustering_dbscan_eps = ui.number(
            label="Eps",
            min=0.0,
            value=0.5,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_dbscan_eps.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_DBSCAN,
        )
        self.clustering_dbscan_min_samples = ui.number(
            label="Min Samples",
            min=1,
            value=5,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_dbscan_min_samples.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_DBSCAN,
        )
        self.clustering_agglomerative_n_clusters = ui.number(
            label="Clusters",
            min=1,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_agglomerative_n_clusters.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_AGGLOMERATIVE,
        )

    def build_draw(self):
        self.draw_3d = ui.toggle(
            options=[DRAW_2D, DRAW_3D],
            value=DRAW_3D,
            on_change=self.update,
        )
        self.draw_iterations = ui.number(
            label="Layout Iterations",
            min=1,
            max=1000,
            value=DRAW_ITERATIONS,
            on_change=self.update,
        )
        self.draw_aggregation = ui.select(
            label="Aggregation",
            options=[
                DRAW_MEAN,
                DRAW_MEDIAN,
                DRAW_MODE,
            ],
            value=DRAW_MEAN,
            on_change=self.update,
        )

    def build_plot(self):
        fig = go.Figure()
        fig.layout.width = None
        fig.layout.autosize = True
        self.plot_container = ui.element("div").classes("w-full h-full")

    async def update(self, _=None):
        self.state.source_type = str(self.data_source_type.value)
        self.state.source_name = str(self.data_source_example_file.value)
        self.state.source_openml = str(self.data_source_openml.value)
        self.state.lens_type = str(self.lens_type.value)
        self.state.lens_pca_n_components = int(self.pca_n_components.value)
        self.state.lens_umap_n_components = int(self.umap_n_components.value)
        self.state.cover_type = str(self.cover_type.value)
        self.state.cover_cubical_n_intervals = int(self.cover_cubical_n_intervals.value)
        self.state.cover_cubical_overlap_frac = float(
            self.cover_cubical_overlap_frac.value
        )
        self.state.cover_knn_neighbors = int(self.cover_knn_neighbors.value)
        self.state.cover_ball_radius = float(self.cover_ball_radius.value)
        self.state.clustering_type = str(self.clustering_type.value)
        self.state.clustering_kmeans_n_clusters = int(
            self.clustering_kmeans_n_clusters.value
        )
        self.state.clustering_dbscan_eps = float(self.clustering_dbscan_eps.value)
        self.state.clustering_dbscan_min_samples = int(
            self.clustering_dbscan_min_samples.value
        )
        self.state.clustering_agglomerative_n_clusters = int(
            self.clustering_agglomerative_n_clusters.value
        )
        self.state.draw_dim = 3 if self.draw_3d.value == DRAW_3D else 2
        self.state.draw_iterations = int(self.draw_iterations.value)
        self.state.draw_aggregation = self.draw_aggregation.value
        await self.render()

    async def render(self):
        mapper_graph, mapper_fig = await run.cpu_bound(
            compute_mapper,
            **asdict(self.state),
        )

        self.plot_container.clear()
        with self.plot_container:
            ui.plotly(mapper_fig)

    async def update_csv_handler(self, file):
        self.csv_file = run.cpu_bound(update_csv, file)
        await self.update_dataset_handler()
        await self.update_plot_handler()

    def __init__(self):
        self.state = State()
        with ui.row().classes("w-full h-screen m-0 p-0 gap-0 overflow-hidden"):
            with ui.column().classes("w-64 h-full m-0 p-0"):
                with ui.column().classes("w-64 h-full overflow-y-auto p-3 gap-2"):
                    with ui.card().classes("w-full"):
                        ui.markdown("#### 📊 Data")
                        self.build_dataset()
                    with ui.card().classes("w-full"):
                        ui.markdown("#### 🔎 Lens")
                        self.build_lens()
                    with ui.card().classes("w-full"):
                        ui.markdown("#### 🌐 Cover")
                        self.build_cover()
                    with ui.card().classes("w-full"):
                        ui.markdown("#### 🧮 Clustering")
                        self.build_clustering()

            with ui.column().classes("flex-1 h-full overflow-hidden m-0 p-0"):
                with ui.row(align_items="baseline"):
                    self.build_draw()
                self.build_plot()
        mapper_graph, mapper_fig = compute_mapper(**asdict(self.state))
        with self.plot_container:
            ui.plotly(mapper_fig)


def main():
    App()
    ui.run()


if __name__ in {"__main__", "__mp_main__", "tdamapper.app"}:
    main()
