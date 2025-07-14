"""
This module provides a web app for visualizing Mapper graphs.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nicegui import app, run, ui
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from tdamapper.core import TrivialClustering
from tdamapper.cover import BallCover, CubicalCover, KNNCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot
from tdamapper.protocols import Clustering, Cover

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GIT_REPO_URL = "https://github.com/lucasimi/tda-mapper-python"

ICON_URL = f"{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-icon.png"

LOGO_URL = f"{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png"

ABOUT_TEXT = """
### About

**tda-mapper** is a Python library built around the Mapper algorithm, a core
technique in Topological Data Analysis (TDA) for extracting topological
structure from complex data. Designed for computational efficiency and
scalability, it leverages optimized spatial search methods to support
high-dimensional datasets. You can find further details in the
[documentation](https://tda-mapper.readthedocs.io/en/main/)
and in the
[paper](https://openreview.net/pdf?id=lTX4bYREAZ).
"""

LOAD_EXAMPLE = "Example"
LOAD_EXAMPLE_DIGITS = "Digits"
LOAD_EXAMPLE_IRIS = "Iris"
LOAD_CSV = "CSV"

LENS_IDENTITY = "Identity"
LENS_PCA = "PCA"
LENS_UMAP = "UMAP"

LENS_PCA_N_COMPONENTS = 2
LENS_UMAP_N_COMPONENTS = 2

COVER_SCALE_DATA = False

COVER_CUBICAL = "Cubical Cover"
COVER_BALL = "Ball Cover"
COVER_KNN = "KNN Cover"

CLUSTERING_TRIVIAL = "Skip"
CLUSTERING_KMEANS = "KMeans"
CLUSTERING_DBSCAN = "DBSCAN"
CLUSTERING_AGGLOMERATIVE = "Agglomerative Clustering"

COVER_CUBICAL_N_INTERVALS = 10
COVER_CUBICAL_OVERLAP_FRAC = 0.25
COVER_KNN_NEIGHBORS = 10
COVER_BALL_RADIUS = 100.0

CLUSTERING_SCALE_DATA = False

CLUSTERING_KMEANS_N_CLUSTERS = 2
CLUSTERING_DBSCAN_EPS = 0.5
CLUSTERING_DBSCAN_MIN_SAMPLES = 5
CLUSTERING_AGGLOMERATIVE_N_CLUSTERS = 2

PLOT_DIMENSIONS: Literal[2, 3] = 2
PLOT_ITERATIONS = 100
PLOT_COLORMAP = "Viridis"
PLOT_NODE_SIZE = 1.0

RANDOM_SEED = 42


@dataclass
class MapperConfig:
    """
    Configuration for the Mapper algorithm.

    :param lens_type: Type of lens to use for dimensionality reduction.
    :param cover_scale_data: Whether to scale the data before covering.
    :param cover_type: Type of cover to use for the Mapper algorithm.
    :param clustering_scale_data: Whether to scale the data before clustering.
    :param clustering_type: Type of clustering algorithm to use.
    :param lens_pca_n_components: Number of components for PCA lens.
    :param lens_umap_n_components: Number of components for UMAP lens.
    :param cover_cubical_n_intervals: Number of intervals for cubical cover.
    :param cover_cubical_overlap_frac: Overlap fraction for cubical cover.
    :param cover_ball_radius: Radius for ball cover.
    :param cover_knn_neighbors: Number of neighbors for KNN cover.
    :param clustering_kmeans_n_clusters: Number of clusters for KMeans
        clustering.
    :param clustering_dbscan_eps: Epsilon parameter for DBSCAN clustering.
    :param clustering_dbscan_min_samples: Minimum samples for DBSCAN
        clustering.
    :param clustering_agglomerative_n_clusters: Number of clusters for
        Agglomerative clustering.
    :param plot_dimensions: Number of dimensions for the plot (2D or 3D).
    :param plot_iterations: Number of iterations for the plot.
    :param plot_seed: Random seed for reproducibility.
    """

    lens_type: str = LENS_PCA
    cover_scale_data: bool = COVER_SCALE_DATA
    cover_type: str = COVER_CUBICAL
    clustering_scale_data: bool = CLUSTERING_SCALE_DATA
    clustering_type: str = CLUSTERING_TRIVIAL
    lens_pca_n_components: int = LENS_PCA_N_COMPONENTS
    lens_umap_n_components: int = LENS_UMAP_N_COMPONENTS
    cover_cubical_n_intervals: int = COVER_CUBICAL_N_INTERVALS
    cover_cubical_overlap_frac: float = COVER_CUBICAL_OVERLAP_FRAC
    cover_ball_radius: float = COVER_BALL_RADIUS
    cover_knn_neighbors: int = COVER_KNN_NEIGHBORS
    clustering_kmeans_n_clusters: int = CLUSTERING_KMEANS_N_CLUSTERS
    clustering_dbscan_eps: float = CLUSTERING_DBSCAN_EPS
    clustering_dbscan_min_samples: int = CLUSTERING_DBSCAN_MIN_SAMPLES
    clustering_agglomerative_n_clusters: int = CLUSTERING_AGGLOMERATIVE_N_CLUSTERS
    plot_dimensions: Literal[2, 3] = PLOT_DIMENSIONS
    plot_iterations: int = PLOT_ITERATIONS
    plot_seed: int = RANDOM_SEED


def fix_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes the input data by selecting numeric columns, dropping empty columns,
    and filling NaN values with the mean of each column.

    :param data: Input DataFrame to be fixed.
    :return: Fixed DataFrame with numeric columns, no empty columns, and NaN
        values filled with column means.
    """
    df = pd.DataFrame(data)
    df = df.select_dtypes(include="number")
    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def lens_identity(X: NDArray[np.float_]) -> NDArray[np.float_]:
    """
    Identity lens function that returns the input data as is.

    :param X: Input data as a NumPy array.
    :return: The same input data as a NumPy array.
    """
    return X


def lens_pca(n_components: int) -> Callable[[NDArray[np.float_]], NDArray[np.float_]]:
    """
    Creates a lens function that reduces the dimensionality of the input data.
    This function applies PCA to the input data and returns the transformed
    data.

    :param n_components: Number of components to keep after PCA.
    :return: A function that applies PCA to the input data and returns the
        transformed data.
    """

    def _pca(X: NDArray[np.float_]) -> NDArray[np.float_]:
        pca_model = PCA(n_components=n_components, random_state=RANDOM_SEED)
        return pca_model.fit_transform(X)

    return _pca


def lens_umap(n_components: int) -> Callable[[NDArray[np.float_]], NDArray[np.float_]]:
    """
    Creates a lens function that reduces the dimensionality of the input data.
    This function applies UMAP to the input data and returns the transformed
    data.

    :param n_components: Number of components to keep after UMAP.
    :return: A function that applies UMAP to the input data and returns the
        transformed data.
    """

    def _umap(X: NDArray[np.float_]) -> NDArray[np.float_]:
        um = UMAP(n_components=n_components, random_state=RANDOM_SEED)
        return um.fit_transform(X)

    return _umap


def run_mapper(
    df: pd.DataFrame, **kwargs: dict[str, Any]
) -> Optional[tuple[nx.Graph, pd.DataFrame]]:
    """
    Runs the Mapper algorithm on the provided DataFrame and returns the Mapper
    graph and the transformed DataFrame.

    :param df: Input DataFrame containing the data to be processed.
    :param kwargs: Additional parameters for the Mapper configuration.
    :return: A tuple containing the Mapper graph and the transformed DataFrame,
        or None if the computation fails.
    """
    logger.info("Mapper computation started...")
    if df is None or df.empty:
        error = "Mapper computation failed: no data found, please load data first."
        logger.error(error)
        return None

    params: dict[str, Any] = kwargs
    mapper_config = MapperConfig(**params)

    lens_type = mapper_config.lens_type
    cover_scale_data = mapper_config.cover_scale_data
    cover_type = mapper_config.cover_type
    clustering_scale_data = mapper_config.clustering_scale_data
    clustering_type = mapper_config.clustering_type
    lens_pca_n_components = mapper_config.lens_pca_n_components
    lens_umap_n_components = mapper_config.lens_umap_n_components
    cover_cubical_n_intervals = mapper_config.cover_cubical_n_intervals
    cover_cubical_overlap_frac = mapper_config.cover_cubical_overlap_frac
    cover_ball_radius = mapper_config.cover_ball_radius
    cover_knn_neighbors = mapper_config.cover_knn_neighbors
    clustering_kmeans_n_clusters = mapper_config.clustering_kmeans_n_clusters
    clustering_dbscan_eps = mapper_config.clustering_dbscan_eps
    clustering_dbscan_min_samples = mapper_config.clustering_dbscan_min_samples
    clustering_agglomerative_n_clusters = (
        mapper_config.clustering_agglomerative_n_clusters
    )

    lens = lens_pca(n_components=LENS_PCA_N_COMPONENTS)
    if lens_type == LENS_IDENTITY:
        lens = lens_identity
    elif lens_type == LENS_PCA:
        lens = lens_pca(n_components=lens_pca_n_components)
    elif lens_type == LENS_UMAP:
        lens = lens_umap(n_components=lens_umap_n_components)

    cover: Cover[NDArray[np.float_]]
    if cover_type == COVER_CUBICAL:
        cover = CubicalCover(
            n_intervals=cover_cubical_n_intervals,
            overlap_frac=cover_cubical_overlap_frac,
        )
    elif cover_type == COVER_BALL:
        cover = BallCover(radius=cover_ball_radius)
    elif cover_type == COVER_KNN:
        cover = KNNCover(neighbors=cover_knn_neighbors)
    else:
        logger.error(f"Unknown cover type: {cover_type}")
        return None

    clustering: Clustering[NDArray[np.float_]]
    if clustering_type == CLUSTERING_TRIVIAL:
        clustering = TrivialClustering()
    elif clustering_type == CLUSTERING_KMEANS:
        clustering = KMeans(
            n_clusters=clustering_kmeans_n_clusters,
            random_state=RANDOM_SEED,
        )
    elif clustering_type == CLUSTERING_DBSCAN:
        clustering = DBSCAN(
            eps=clustering_dbscan_eps,
            min_samples=clustering_dbscan_min_samples,
        )
    elif clustering_type == CLUSTERING_AGGLOMERATIVE:
        clustering = AgglomerativeClustering(
            n_clusters=clustering_agglomerative_n_clusters,
        )
    else:
        logger.error(f"Unknown clustering type: {clustering_type}")
        return None

    mapper = MapperAlgorithm(cover=cover, clustering=clustering)
    X = df.to_numpy()
    y = lens(X)
    df_y = pd.DataFrame(y, columns=[f"{lens_type} {i}" for i in range(y.shape[1])])
    if cover_scale_data:
        y = StandardScaler().fit_transform(y)
    if clustering_scale_data:
        X = StandardScaler().fit_transform(X)
    mapper_graph = mapper.fit_transform(X, y)
    logger.info("Mapper computation completed.")
    return mapper_graph, df_y


def create_mapper_figure(
    df_X: pd.DataFrame,
    df_y: pd.DataFrame,
    df_target: pd.DataFrame,
    mapper_graph: nx.Graph,
    **kwargs: dict[str, Any],
) -> go.Figure:
    """
    Renders the Mapper graph as a Plotly figure.

    :param df_X: DataFrame containing the input data.
    :param df_y: DataFrame containing the lens-transformed data.
    :param df_target: DataFrame containing the target labels.
    :param mapper_graph: The Mapper graph to be visualized.
    :param kwargs: Additional parameters for the Mapper configuration.
    :return: A Plotly figure representing the Mapper graph.
    """
    logger.info("Mapper rendering started...")
    df_colors = pd.concat([df_target, df_y, df_X], axis=1)
    params: dict[str, Any] = kwargs
    mapper_config = MapperConfig(**params)
    plot_dimensions = mapper_config.plot_dimensions
    plot_iterations = mapper_config.plot_iterations
    plot_seed = mapper_config.plot_seed
    mapper_fig = MapperPlot(
        mapper_graph,
        dim=plot_dimensions,
        iterations=plot_iterations,
        seed=plot_seed,
    ).plot_plotly(
        colors=df_colors.to_numpy(),
        title=df_colors.columns.to_list(),
        cmap=[
            PLOT_COLORMAP,
            "Cividis",
            "Jet",
            "Plasma",
            "RdBu",
        ],
        height=800,
        node_size=[i * 0.125 * PLOT_NODE_SIZE for i in range(17)],
    )
    mapper_fig.update_layout(
        width=None,
        height=None,
        autosize=True,
        xaxis=dict(scaleanchor="y"),
        uirevision="constant",
    )
    logger.info("Mapper rendering completed.")
    return mapper_fig


class App:
    """
    Main application class for the Mapper web application.

    This class initializes the user interface, handles data loading, and runs
    the Mapper algorithm.

    :param storage: Dictionary to store application state and data.
    :param draw_area: Optional draw area for rendering Mapper graphs.
    :param plot_container: Container for the plot area.
    :param left_drawer: Drawer for the left sidebar containing controls and
        settings.
    :param lens_type: Type of lens to use for dimensionality reduction.
    :param cover_type: Type of cover to use for the Mapper algorithm.
    :param clustering_type: Type of clustering algorithm to use.
    :param lens_pca_n_components: Number of components for PCA lens.
    :param lens_umap_n_components: Number of components for UMAP lens.
    :param cover_cubical_n_intervals: Number of intervals for cubical cover.
    :param cover_cubical_overlap_frac: Overlap fraction for cubical cover.
    :param cover_ball_radius: Radius for ball cover.
    :param cover_knn_neighbors: Number of neighbors for KNN cover.
    :param clustering_kmeans_n_clusters: Number of clusters for KMeans
        clustering.
    :param clustering_dbscan_eps: Epsilon parameter for DBSCAN clustering.
    :param clustering_dbscan_min_samples: Minimum samples for DBSCAN
        clustering.
    :param clustering_agglomerative_n_clusters: Number of clusters for
        Agglomerative clustering.
    :param plot_dimensions: Number of dimensions for the plot (2D or 3D).
    :param plot_iterations: Number of iterations for the plot.
    :param plot_seed: Random seed for reproducibility.
    :param load_type: Type of data loading (example or CSV).
    :param load_example: Example dataset to load if using example data.
    """

    lens_type: Any
    cover_type: Any
    clustering_type: Any
    lens_pca_n_components: Any
    lens_umap_n_components: Any
    cover_cubical_n_intervals: Any
    cover_cubical_overlap_frac: Any
    cover_ball_radius: Any
    cover_knn_neighbors: Any
    clustering_kmeans_n_clusters: Any
    clustering_dbscan_eps: Any
    clustering_dbscan_min_samples: Any
    clustering_agglomerative_n_clusters: Any
    plot_dimensions: Any
    plot_iterations: Any
    plot_seed: Any
    load_type: Any
    load_example: Any
    storage: dict[str, Any]
    draw_area: Optional[Any] = None
    plot_container: Any
    left_drawer: Any

    def __init__(self, storage: dict[str, Any]) -> None:
        self.storage = storage

        ui.colors(
            themelight="#ebedf8",
            themedark="#132f48",
        )

        self.left_drawer = ui.left_drawer(elevated=True).classes(
            "w-96 h-full overflow-y-auto gap-12"
        )

        with self.left_drawer:
            with ui.link(target=GIT_REPO_URL, new_tab=True).classes("w-full"):
                ui.image(LOGO_URL)

            with ui.column().classes("w-full gap-2"):
                self._init_about()

            with ui.column().classes("w-full gap-2"):
                self._init_file_upload()

            ui.button(
                "⬆️ Load Data",
                on_click=self.load_data,
                color="themelight",
            ).classes("w-full text-themedark")

            with ui.column().classes("w-full gap-2"):
                self._init_lens()

            with ui.column().classes("w-full gap-2"):
                self._init_cover()

            with ui.column().classes("w-full gap-2"):
                self._init_clustering()

            ui.button(
                "🚀 Run Mapper",
                on_click=self.async_run_mapper,
                color="themelight",
            ).classes("w-full text-themedark")

            with ui.column().classes("w-full gap-2"):
                self._init_draw()

            ui.button(
                "🌊 Redraw",
                on_click=self.async_draw_mapper,
                color="themelight",
            ).classes("w-full text-themedark")

            with ui.column().classes("w-full gap-2"):
                self._init_footnotes()

        with ui.column().classes("w-full h-screen overflow-hidden"):
            self._init_draw_area()

    def _init_about(self) -> None:
        with ui.dialog() as dialog, ui.card():
            ui.markdown(ABOUT_TEXT)
            ui.link(
                text="If you like this project, please consider giving it a ⭐ on GitHub!",
                target=GIT_REPO_URL,
                new_tab=True,
            ).classes("w-full")
            ui.button("Close", on_click=dialog.close, color="themelight").classes(
                "w-full text-themedark"
            )
        ui.button("ℹ️ About", on_click=dialog.open, color="themelight").classes(
            "w-full text-themedark"
        )

    def _init_file_upload(self) -> None:
        ui.label("📊 Data").classes("text-h6")

        self.load_type = ui.select(
            options=[LOAD_EXAMPLE, LOAD_CSV],
            label="Data Source",
            value=LOAD_EXAMPLE,
        ).classes("w-full")

        upload = ui.upload(
            on_upload=self.upload_file,
            auto_upload=True,
            label="Upload CSV File",
        ).classes("w-full mt-4")
        upload.props("accept=.csv")
        upload.bind_visibility_from(
            target_object=self.load_type,
            target_name="value",
            value=LOAD_CSV,
        )

        self.load_example = ui.select(
            options=[LOAD_EXAMPLE_DIGITS, LOAD_EXAMPLE_IRIS],
            label="Dataset",
            value=LOAD_EXAMPLE_DIGITS,
        ).classes("w-full")
        self.load_example.bind_visibility_from(
            target_object=self.load_type,
            target_name="value",
            value=LOAD_EXAMPLE,
        )

    def _init_lens(self) -> None:
        ui.label("🔎 Lens").classes("text-h6")
        self._init_lens_settings()

    def _init_lens_settings(self) -> None:
        self.lens_type = ui.select(
            options=[
                LENS_IDENTITY,
                LENS_PCA,
                LENS_UMAP,
            ],
            label="Lens",
            value=LENS_PCA,
        ).classes("w-full")

        self.lens_pca_n_components = ui.number(
            label="PCA Components",
            value=LENS_PCA_N_COMPONENTS,
        ).classes("w-full")
        self.lens_pca_n_components.bind_visibility_from(
            target_object=self.lens_type,
            target_name="value",
            value=LENS_PCA,
        )

        self.lens_umap_n_components = ui.number(
            label="UMAP Components",
            value=LENS_UMAP_N_COMPONENTS,
        ).classes("w-full")
        self.lens_umap_n_components.bind_visibility_from(
            target_object=self.lens_type,
            target_name="value",
            value=LENS_UMAP,
        )

    def _init_cover(self) -> None:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("🌐 Cover").classes("text-h6")
            self.cover_scale = ui.switch(
                text="Scaling",
                value=COVER_SCALE_DATA,
            )
        self._init_cover_settings()

    def _init_cover_settings(self) -> None:
        self.cover_type = ui.select(
            options=[
                COVER_CUBICAL,
                COVER_BALL,
                COVER_KNN,
            ],
            label="Cover",
            value=COVER_CUBICAL,
        ).classes("w-full")

        self.cover_cubical_n_intervals = ui.number(
            label="Number of Intervals",
            value=COVER_CUBICAL_N_INTERVALS,
        ).classes("w-full")
        self.cover_cubical_n_intervals.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_CUBICAL,
        )

        self.cover_cubical_overlap_frac = ui.number(
            label="Overlap Fraction",
            value=COVER_CUBICAL_OVERLAP_FRAC,
        ).classes("w-full")
        self.cover_cubical_overlap_frac.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_CUBICAL,
        )

        self.cover_ball_radius = ui.number(
            label="Ball Radius",
            value=COVER_BALL_RADIUS,
        ).classes("w-full")
        self.cover_ball_radius.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_BALL,
        )

        self.cover_knn_neighbors = ui.number(
            label="Number of Neighbors",
            value=COVER_KNN_NEIGHBORS,
        ).classes("w-full")
        self.cover_knn_neighbors.bind_visibility_from(
            target_object=self.cover_type,
            target_name="value",
            value=COVER_KNN,
        )

    def _init_clustering(self) -> None:
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("🧮 Clustering").classes("text-h6")
            self.clustering_scale = ui.switch(
                text="Scaling",
                value=CLUSTERING_SCALE_DATA,
            )
        self._init_clustering_settings()

    def _init_clustering_settings(self) -> None:
        self.clustering_type = ui.select(
            options=[
                CLUSTERING_TRIVIAL,
                CLUSTERING_KMEANS,
                CLUSTERING_DBSCAN,
                CLUSTERING_AGGLOMERATIVE,
            ],
            label="Clustering",
            value=CLUSTERING_TRIVIAL,
        ).classes("w-full")

        self.clustering_kmeans_n_clusters = ui.number(
            label="Number of Clusters",
            value=CLUSTERING_KMEANS_N_CLUSTERS,
        ).classes("w-full")
        self.clustering_kmeans_n_clusters.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_KMEANS,
        )

        self.clustering_dbscan_eps = ui.number(
            label="Epsilon",
            value=CLUSTERING_DBSCAN_EPS,
        ).classes("w-full")
        self.clustering_dbscan_eps.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_DBSCAN,
        )
        self.clustering_dbscan_min_samples = ui.number(
            label="Min Samples",
            value=CLUSTERING_DBSCAN_MIN_SAMPLES,
        ).classes("w-full")
        self.clustering_dbscan_min_samples.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_DBSCAN,
        )

        self.clustering_agglomerative_n_clusters = ui.number(
            label="Number of Clusters",
            value=CLUSTERING_AGGLOMERATIVE_N_CLUSTERS,
        ).classes("w-full")
        self.clustering_agglomerative_n_clusters.bind_visibility_from(
            target_object=self.clustering_type,
            target_name="value",
            value=CLUSTERING_AGGLOMERATIVE,
        )

    def _init_draw(self) -> None:
        ui.label("🎨 Draw").classes("text-h6")
        self._init_draw_settings()

    def _init_draw_settings(self) -> None:
        self.plot_dimensions = ui.select(
            options=[2, 3],
            label="Dimensions",
            value=PLOT_DIMENSIONS,
        ).classes("w-full")
        self.plot_iterations = ui.number(
            label="Iterations",
            value=PLOT_ITERATIONS,
            min=1,
            max=10 * PLOT_ITERATIONS,
        ).classes("w-full")
        self.plot_seed = ui.number(
            label="Seed",
            value=RANDOM_SEED,
        ).classes("w-full")

    def _init_footnotes(self) -> None:
        ui.label(text="Made in Rome, with ❤️ and ☕️.").classes(
            "text-caption text-gray-500"
        ).classes("text-caption text-gray-500")

    def _init_draw_area(self) -> None:
        self.plot_container = ui.element("div").classes("w-full h-full")
        self.draw_area = None

        def _toggle_drawer() -> None:
            self.left_drawer.toggle()

        with ui.page_sticky(x_offset=18, y_offset=18, position="top-left"):
            toggle_button = ui.button(
                icon="menu",
                on_click=_toggle_drawer,
            ).props("fab color=themedark")
            toggle_button.bind_visibility_from(
                target_object=self.left_drawer,
                target_name="value",
                value=False,
            )

    def get_mapper_config(self) -> MapperConfig:
        """
        Retrieves the current configuration settings for the Mapper algorithm.

        :return: A MapperConfig object containing the current settings.
        """
        plot_dim = int(self.plot_dimensions.value)
        plot_dimensions: Literal[2, 3]
        if plot_dim == 2:
            plot_dimensions = 2
        elif plot_dim == 3:
            plot_dimensions = 3
        else:
            plot_dimensions = PLOT_DIMENSIONS
        return MapperConfig(
            lens_type=str(self.lens_type.value) if self.lens_type.value else LENS_PCA,
            cover_type=(
                str(self.cover_type.value) if self.cover_type.value else COVER_CUBICAL
            ),
            clustering_type=(
                str(self.clustering_type.value)
                if self.clustering_type.value
                else CLUSTERING_TRIVIAL
            ),
            lens_pca_n_components=(
                int(self.lens_pca_n_components.value)
                if self.lens_pca_n_components.value
                else LENS_PCA_N_COMPONENTS
            ),
            lens_umap_n_components=(
                int(self.lens_umap_n_components.value)
                if self.lens_umap_n_components.value
                else LENS_UMAP_N_COMPONENTS
            ),
            cover_cubical_n_intervals=(
                int(self.cover_cubical_n_intervals.value)
                if self.cover_cubical_n_intervals.value
                else COVER_CUBICAL_N_INTERVALS
            ),
            cover_cubical_overlap_frac=(
                float(self.cover_cubical_overlap_frac.value)
                if self.cover_cubical_overlap_frac.value
                else COVER_CUBICAL_OVERLAP_FRAC
            ),
            cover_ball_radius=(
                float(self.cover_ball_radius.value)
                if self.cover_ball_radius.value
                else COVER_BALL_RADIUS
            ),
            cover_knn_neighbors=(
                int(self.cover_knn_neighbors.value)
                if self.cover_knn_neighbors.value
                else COVER_KNN_NEIGHBORS
            ),
            clustering_kmeans_n_clusters=(
                int(self.clustering_kmeans_n_clusters.value)
                if self.clustering_kmeans_n_clusters.value
                else CLUSTERING_KMEANS_N_CLUSTERS
            ),
            clustering_dbscan_eps=(
                float(self.clustering_dbscan_eps.value)
                if self.clustering_dbscan_eps.value
                else CLUSTERING_DBSCAN_EPS
            ),
            clustering_dbscan_min_samples=(
                int(self.clustering_dbscan_min_samples.value)
                if self.clustering_dbscan_min_samples.value
                else CLUSTERING_DBSCAN_MIN_SAMPLES
            ),
            clustering_agglomerative_n_clusters=(
                int(self.clustering_agglomerative_n_clusters.value)
                if self.clustering_agglomerative_n_clusters.value
                else CLUSTERING_AGGLOMERATIVE_N_CLUSTERS
            ),
            plot_dimensions=plot_dimensions,
            plot_iterations=(
                int(self.plot_iterations.value)
                if self.plot_iterations.value
                else PLOT_ITERATIONS
            ),
            plot_seed=(
                int(self.plot_seed.value) if self.plot_seed.value else RANDOM_SEED
            ),
        )

    def upload_file(self, file: Any) -> None:
        """
        Handles the file upload event, reads the CSV file,
        and stores the data in the application storage.

        :param file: The uploaded file object.
        :return: None
        """
        if file is not None:
            df = pd.read_csv(file.content)
            self.storage["df"] = fix_data(df)
            self.storage["labels"] = pd.DataFrame()
            message = "File upload completed."
            logger.info(message)
            ui.notify(message, type="info")
        else:
            error = "File upload failed: no file provided."
            logger.info(error)
            ui.notify(error, type="warning")

    def load_data(self) -> None:
        """
        Loads example datasets or CSV files based on the selected load type.

        If the load type is set to "Example", it loads either the Digits or
        Iris dataset. If the load type is set to "CSV", it checks if a
        DataFrame is already stored in the application storage and uses it.

        :return: None
        """
        if self.load_type.value == LOAD_EXAMPLE:
            if self.load_example.value == LOAD_EXAMPLE_DIGITS:
                df, labels = load_digits(as_frame=True, return_X_y=True)
            elif self.load_example.value == LOAD_EXAMPLE_IRIS:
                df, labels = load_iris(as_frame=True, return_X_y=True)
            else:
                error = "Load data failed: unknown example dataset selected."
                logger.error(error)
                ui.notify(error, type="warning")
                return
            self.storage["df"] = fix_data(df)
            self.storage["labels"] = fix_data(labels)
        elif self.load_type.value == LOAD_CSV:
            df = self.storage.get("df", pd.DataFrame())
            if df is None or df.empty:
                error = "Load data failed: no data found, please upload a file first."
                logger.warning(error)
                ui.notify(error, type="warning")
                return
        else:
            error = "Load data failed: unknown load type selected."
            logger.error(error)
            ui.notify(error, type="warning")
            return

        df = self.storage.get("df", pd.DataFrame())
        if df is not None and not df.empty:
            logger.info("Load data completed.")
            ui.notify("Load data completed.", type="positive")
        else:
            error = "Load data failed: no data found, please upload a file first."
            logger.warning(error)
            ui.notify(error, type="warning")

    def notification_running_start(self, message: str) -> Any:
        """
        Starts a notification to indicate that a long-running operation is in
        progress.

        :param message: The message to display in the notification.
        :return: A notification object that can be used to update the message
            and status.
        """
        notification = ui.notification(timeout=None, type="ongoing")
        notification.message = message
        notification.spinner = True
        return notification

    def notification_running_stop(
        self, notification: Any, message: str, type: Optional[str] = None
    ) -> None:
        """
        Stops the notification and updates it with the final message and type.

        :param notification: The notification object to update.
        :param message: The final message to display in the notification.
        :param type: The type of notification.
        :return: None
        """
        if type is not None:
            notification.type = type
        notification.message = message
        notification.timeout = 5.0
        notification.spinner = False

    async def async_run_mapper(self) -> None:
        """
        Runs the Mapper algorithm on the loaded data and updates the storage
        with the Mapper graph and transformed DataFrame.

        This method retrieves the input DataFrame from storage, applies the
        Mapper algorithm, and stores the resulting Mapper graph and transformed
        DataFrame back into storage.

        :return: None
        """
        notification = self.notification_running_start("Running Mapper...")
        df_X = self.storage.get("df", pd.DataFrame())
        if df_X is None or df_X.empty:
            error = "Run Mapper failed: no data found, please load data first."
            logger.warning(error)
            self.notification_running_stop(notification, error, type="warning")
            return
        mapper_config = self.get_mapper_config()
        result = await run.cpu_bound(run_mapper, df_X, **asdict(mapper_config))
        if result is None:
            error = "Run Mapper failed: something went wrong."
            logger.error(error)
            self.notification_running_stop(notification, error, type="error")
            return
        mapper_graph, df_y = result
        if mapper_graph is not None:
            self.storage["mapper_graph"] = mapper_graph
        if df_y is not None:
            self.storage["df_y"] = df_y
        self.notification_running_stop(
            notification, "Run Mapper completed.", type="positive"
        )
        await self.async_draw_mapper()

    async def async_draw_mapper(self) -> None:
        """
        Draws the Mapper graph using the stored graph and input data.

        This method retrieves the Mapper graph and input DataFrame from
        storage, creates a Plotly figure representing the Mapper graph, and
        updates the draw area in the user interface with the new figure.

        :return: None
        """
        notification = self.notification_running_start("Drawing Mapper...")

        mapper_config = self.get_mapper_config()

        df_X = self.storage.get("df", pd.DataFrame())
        df_y = self.storage.get("df_y", pd.DataFrame())
        df_target = self.storage.get("labels", pd.DataFrame())
        mapper_graph = self.storage.get("mapper_graph", None)

        if df_X.empty or mapper_graph is None:
            error = (
                "Draw Mapper failed: no Mapper graph found, please run Mapper first."
            )
            logger.warning(error)
            self.notification_running_stop(
                notification=notification, message=error, type="warning"
            )
            return

        mapper_fig = await run.cpu_bound(
            create_mapper_figure,
            df_X,
            df_y,
            df_target,
            mapper_graph,
            **asdict(mapper_config),
        )
        if mapper_fig is None:
            error = "Draw Mapper failed: something went wrong."
            self.notification_running_stop(notification, error, type="error")
            return

        logger.info("Displaying Mapper plot.")
        if self.draw_area is not None:
            self.draw_area.clear()
            self.plot_container.clear()
        with self.plot_container:
            self.draw_area = ui.plotly(mapper_fig).classes("w-full h-full")
        self.notification_running_stop(
            notification, "Draw Mapper completed.", type="positive"
        )


def startup() -> None:
    """
    Initializes the NiceGUI app and sets up the main page.

    :return: None
    """

    @ui.page("/")
    def main_page() -> None:
        """
        Main page of the application.

        :return: None
        """
        ui.query(".nicegui-content").classes("p-0")
        storage = app.storage.client
        App(storage=storage)


def main() -> None:
    """
    Main entry point for the Mapper web application.

    :return: None
    """
    port = os.getenv("PORT", "8080")
    host = os.getenv("HOST", "0.0.0.0")
    production = os.getenv("PRODUCTION", "false").lower() == "true"
    storage_secret = os.getenv("STORAGE_SECRET", "storage_secret")
    startup()
    ui.run(
        storage_secret=storage_secret,
        reload=not production,
        host=host,
        title="tda-mapper-app",
        favicon=ICON_URL,
        port=int(port),
    )
