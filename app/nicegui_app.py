import numpy as np
import pandas as pd
import plotly.graph_objs as go
from nicegui import run, ui
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from umap import UMAP

from tdamapper.core import TrivialClustering, TrivialCover
from tdamapper.cover import BallCover, CubicalCover, KNNCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot


def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    mode_values = values[counts == max_count]
    return np.nanmean(mode_values)


def _identity(X):
    return X


def _pca(n_components):
    pca = PCA(n_components=n_components, random_state=42)

    def _func(X):
        return pca.fit_transform(X)

    return _func


def _umap(n_components):
    um = UMAP(n_components=n_components, random_state=42)

    def _func(X):
        return um.fit_transform(X)

    return _func


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

DRAW_3D = "3D"
DRAW_2D = "2D"
DRAW_ITERATIONS = 50


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
            on_change=self.update_dataset_handler,
        ).classes("w-full")
        self.data_source_example_file = ui.select(
            label="File",
            options=[
                DATA_SOURCE_EXAMPLE_DIGITS,
                DATA_SOURCE_EXAMPLE_IRIS,
            ],
            value=DATA_SOURCE_EXAMPLE_DIGITS,
            on_change=self.update_dataset_handler,
        ).classes("w-full")
        self.data_source_example_file.bind_visibility_from(
            target_object=self.data_source_type,
            target_name="value",
            value=DATA_SOURCE_EXAMPLE,
        )
        self.data_source_csv = ui.upload(
            on_upload=self.update_dataset_handler,
        ).classes("w-full")
        self.data_source_csv.bind_visibility_from(
            target_object=self.data_source_type,
            target_name="value",
            value=DATA_SOURCE_CSV,
        )
        self.data_source_openml = ui.input(
            label="OpenML Code",
            on_change=self.update_dataset_handler,
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
            on_change=self.update_graph_handler,
        ).classes("w-full")
        self.pca_n_components = ui.number(
            label="PCA Components",
            min=1,
            max=10,
            value=2,
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
        ).classes("w-full")
        self.cover_cubical_n_intervals = ui.number(
            label="Intervals",
            min=1,
            max=100,
            value=2,
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
        ).classes("w-full")
        self.clustering_kmeans_n_clusters = ui.number(
            label="Clusters",
            min=1,
            value=2,
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
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
            on_change=self.update_graph_handler,
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
            on_change=self.update_plot_handler,
        )
        self.draw_iterations = ui.number(
            label="Layout Iterations",
            min=1,
            max=1000,
            value=DRAW_ITERATIONS,
            on_change=self.update_plot_handler,
        )

    def build_plot(self):
        fig = go.Figure()
        fig.layout.width = None
        fig.layout.autosize = True
        self.plot_container = ui.element("div").classes("w-full h-full")
        with self.plot_container:
            ui.plotly(go.Figure())

    def render_dataset(self):
        source_type = self.data_source_type.value
        if source_type == DATA_SOURCE_EXAMPLE:
            name = self.data_source_example_file.value
            if name == DATA_SOURCE_EXAMPLE_DIGITS:
                X, y = load_digits(return_X_y=True, as_frame=True)
                return X, y
            elif name == DATA_SOURCE_EXAMPLE_IRIS:
                X, y = load_iris(return_X_y=True, as_frame=True)
                return X, y
        elif source_type == DATA_SOURCE_CSV:
            pass

    def render_lens(self):
        if self.lens_type.value == LENS_IDENTITY:
            return _identity
        elif self.lens_type.value == LENS_PCA:
            n = int(self.pca_n_components.value)
            return _pca(n)
        elif self.lens_type.value == LENS_UMAP:
            n = int(self.umap_n_components.value)
            return _umap(n)

    def render_cover(self):
        if self.cover_type.value == COVER_TRIVIAL:
            return TrivialCover()
        elif self.cover_type.value == COVER_BALL:
            radius = float(self.cover_ball_radius.value)
            return BallCover(radius=radius)
        elif self.cover_type.value == COVER_CUBICAL:
            n_intervals = int(self.cover_cubical_n_intervals.value)
            overlap_frac = float(self.cover_cubical_overlap_frac.value)
            return CubicalCover(n_intervals=n_intervals, overlap_frac=overlap_frac)
        elif self.cover_type.value == COVER_KNN:
            neighbors = int(self.cover_knn_neighbors.value)
            return KNNCover(neighbors=neighbors)

    def render_clustering(self):
        if self.clustering_type.value == CLUSTERING_TRIVIAL:
            return TrivialClustering()
        elif self.clustering_type.value == CLUSTERING_KMEANS:
            n_clusters = int(self.clustering_kmeans_n_clusters.value)
            return KMeans(n_clusters)
        elif self.clustering_type.value == CLUSTERING_DBSCAN:
            eps = float(self.clustering_dbscan_eps.value)
            min_samples = int(self.clustering_dbscan_min_samples.value)
            return DBSCAN(eps=eps, min_samples=min_samples)
        elif self.clustering_type == CLUSTERING_AGGLOMERATIVE:
            n_clusters = int(self.clustering_agglomerative_n_clusters.value)
            return AgglomerativeClustering(n_clusters=n_clusters)

    async def update_graph_handler(self, _=None):
        await run.io_bound(self.update_graph)

    async def update_dataset_handler(self, _=None):
        await run.io_bound(self.update_dataset)

    def update_dataset(self, _=None):
        self.X, self.labels = self.render_dataset()
        self.update_graph()

    def update_graph(self, _=None):
        self.lens = self.render_lens()
        if self.lens is None:
            return
        if self.X is None:
            return
        self.y = self.lens(self.X)
        cover = self.render_cover()
        if cover is None:
            return
        clustering = self.render_clustering()
        if clustering is None:
            return
        mapper_algo = MapperAlgorithm(
            cover=cover,
            clustering=clustering,
            verbose=False,
        )
        self.mapper_graph = mapper_algo.fit_transform(self.X, self.y)
        self.update_plot()

    async def update_plot_handler(self, _=None):
        await run.io_bound(self.update_plot)

    def update_plot(self):
        if self.mapper_graph is None:
            return

        dim = 3
        if self.draw_3d.value == DRAW_3D:
            dim = 3
        elif self.draw_3d.value == DRAW_2D:
            dim = 2

        iterations = int(self.draw_iterations.value)
        mapper_plot = MapperPlot(
            self.mapper_graph,
            dim=dim,
            iterations=iterations,
            seed=42,
        )
        colors = pd.concat([self.labels, self.X], axis=1)
        colors_arr = colors.to_numpy()
        color_names = colors.columns.tolist()
        mapper_fig = mapper_plot.plot_plotly(
            colors=colors_arr,
            cmap=["jet", "viridis", "cividis"],
            agg=mode,
            title=color_names,
            width=800,
            height=800,
            node_size=list(0.125 * x for x in range(17)),
        )
        mapper_fig.layout.width = None
        mapper_fig.layout.autosize = True
        self.plot_container.clear()
        with self.plot_container:
            ui.plotly(mapper_fig)

    def __init__(self):
        with ui.row().classes("w-full h-screen m-0 p-0 gap-0 overflow-hidden"):
            with ui.column().classes("w-64 h-full m-0 p-0"):  # fixed-width sidebar
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
        self.update_dataset()


app = App()
ui.run()
