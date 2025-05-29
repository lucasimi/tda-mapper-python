import numpy as np
import plotly.graph_objs as go
from nicegui import ui
from nicegui.events import ValueChangeEventArguments
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, make_circles
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


class App:

    def build_lens(self):
        self.opt_lens_id = "Identity"
        self.opt_lens_pca = "PCA"
        self.opt_lens_umap = "UMAP"

        self.lens_type = ui.select(
            label="Lens type",
            options=[
                self.opt_lens_id,
                self.opt_lens_pca,
                self.opt_lens_umap,
            ],
            value=self.opt_lens_pca,
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
            value=self.opt_lens_pca,
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
            value=self.opt_lens_umap,
        )

    def build_cover(self):
        self.opt_cover_trivial = "Trivial"
        self.opt_cover_cubical = "Cubical"
        self.opt_cover_ball = "Ball"
        self.opt_cover_knn = "KNN"

        self.cover_type = ui.select(
            label="Cover type",
            options=[
                self.opt_cover_trivial,
                self.opt_cover_cubical,
                self.opt_cover_ball,
                self.opt_cover_knn,
            ],
            value=self.opt_cover_cubical,
            on_change=self.update,
        ).classes("w-full")
        self.cover_cubical_n = ui.number(
            label="Intervals",
            min=1,
            max=10,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.cover_cubical_n.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=self.opt_cover_cubical,
        )
        self.cover_cubical_overlap = ui.number(
            label="Overlap",
            min=0.0,
            max=1.0,
            value=0.5,
            on_change=self.update,
        ).classes("w-full")
        self.cover_cubical_overlap.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=self.opt_cover_cubical,
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
            value=self.opt_cover_ball,
        )
        self.cover_knn_k = ui.number(
            label="Neighbors",
            min=0,
            value=10,
            on_change=self.update,
        ).classes("w-full")
        self.cover_knn_k.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=self.opt_cover_knn,
        )

    def build_clustering(self):
        self.opt_clustering_trivial = "Trivial"
        self.opt_clustering_kmeans = "KMeans"
        self.opt_clustering_agg = "Agglomerative"
        self.opt_clustering_dbscan = "DBSCAN"

        self.clustering_type = ui.select(
            label="Clustering type",
            options=[
                self.opt_clustering_trivial,
                self.opt_clustering_kmeans,
                self.opt_clustering_agg,
                self.opt_clustering_dbscan,
            ],
            value=self.opt_clustering_trivial,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_kmeans_k = ui.number(
            label="Clusters",
            min=1,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_kmeans_k.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=self.opt_clustering_kmeans,
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
            value=self.opt_clustering_dbscan,
        )
        self.clustering_dbscan_min_samples = ui.number(
            label="Min Samples",
            min=1,
            value=5,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_dbscan_eps.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=self.opt_clustering_dbscan,
        )
        self.clustering_agg_n = ui.number(
            label="Clusters",
            min=1,
            value=2,
            on_change=self.update,
        ).classes("w-full")
        self.clustering_agg_n.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=self.opt_clustering_agg,
        )

    def build_plot(self):
        self.plot = ui.plotly(go.Figure())

    def render_lens(self):
        print(f"Lens type: {self.lens_type.value}")
        if self.lens_type.value == self.opt_lens_id:
            return _identity
        elif self.lens_type.value == self.opt_lens_pca:
            n = int(self.pca_n_components.value)
            return _pca(n)
        elif self.lens_type.value == self.opt_lens_umap:
            n = int(self.umap_n_components.value)
            return _umap(n)

    def render_cover(self):
        if self.cover_type.value == self.opt_cover_trivial:
            return TrivialCover()
        elif self.cover_type.value == self.opt_cover_ball:
            r = float(self.cover_ball_radius.value)
            return BallCover(radius=r)
        elif self.cover_type.value == self.opt_cover_cubical:
            n = int(self.cover_cubical_n.value)
            overlap = float(self.cover_cubical_overlap.value)
            return CubicalCover(n_intervals=n, overlap_frac=overlap)
        elif self.cover_type.value == self.opt_cover_knn:
            k = int(self.cover_knn_k.value)
            return KNNCover(neighbors=k)

    def render_clustering(self):
        if self.clustering_type.value == self.opt_clustering_trivial:
            return TrivialClustering()
        elif self.clustering_type.value == self.opt_clustering_kmeans:
            k = int(self.clustering_kmeans_k.value)
            return KMeans(k)
        elif self.clustering_type.value == self.opt_clustering_dbscan:
            eps = float(self.clustering_dbscan_eps.value)
            min_samples = int(self.clustering_dbscan_min_samples.value)
            return DBSCAN(eps=eps)

    def update(self, _=None):
        X, labels = load_digits(return_X_y=True)
        lens = self.render_lens()
        if lens is None:
            print("Lens is None")
            return
        y = lens(X)

        cover = self.render_cover()
        if cover is None:
            print("Cover is None")
            return

        clustering = self.render_clustering()
        if clustering is None:
            print("Clustering is None")
            return

        mapper_algo = MapperAlgorithm(
            cover=cover,
            clustering=clustering,
            verbose=False,
        )

        mapper_graph = mapper_algo.fit_transform(X, y)

        mapper_plot = MapperPlot(mapper_graph, dim=3, iterations=400, seed=42)

        mapper_fig = mapper_plot.plot_plotly(
            colors=labels,
            cmap=["jet", "viridis", "cividis"],
            agg=mode,
            title="mode of digits",
            width=800,
            height=800,
            node_size=0.5,
        )
        if mapper_fig.layout.width is not None:
            mapper_fig.layout.width = None
        if not mapper_fig.layout.autosize:
            mapper_fig.layout.autosize = True
        mapper_fig.layout.autosize = True
        self.plot.update_figure(mapper_fig)

    def build(self):
        with ui.left_drawer().classes("w-[400px]"):
            self.build_lens()
            ui.separator()
            self.build_cover()
            ui.separator()
            self.build_clustering()
        self.build_plot()
        self.update()


app = App()
app.build()
ui.run()
