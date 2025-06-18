import logging
import os
from dataclasses import asdict, dataclass

import pandas as pd
from nicegui import app, run, ui
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from tdamapper.clustering import TrivialClustering
from tdamapper.cover import BallCover, CubicalCover, KNNCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GIT_REPO_URL = "https://github.com/lucasimi/tda-mapper-python"

ICON_URL = f"{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-icon.png"

LOGO_URL = f"{GIT_REPO_URL}/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png"


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

PLOT_DIMENSIONS = 2
PLOT_ITERATIONS = 100
PLOT_COLORMAP = "Viridis"
PLOT_NODE_SIZE = 1.0

RANDOM_SEED = 42


@dataclass
class MapperConfig:
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
    plot_dimensions: int = PLOT_DIMENSIONS
    plot_iterations: int = PLOT_ITERATIONS
    plot_seed: int = RANDOM_SEED


def fix_data(data):
    df = pd.DataFrame(data)
    df = df.select_dtypes(include="number")
    df.dropna(axis=1, how="all", inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def lens_identity(X):
    return X


def lens_pca(n_components):

    def _pca(X):
        pca_model = PCA(n_components=n_components, random_state=RANDOM_SEED)
        return pca_model.fit_transform(X)

    return _pca


def lens_umap(n_components):

    def _umap(X):
        um = UMAP(n_components=n_components, random_state=RANDOM_SEED)
        return um.fit_transform(X)

    return _umap


def run_mapper(df, **kwargs):
    if df is None or df.empty:
        logger.error("No data found. Please upload a file first.")
        return
    logger.info("Computing Mapper.")

    mapper_config = MapperConfig(**kwargs)

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
        return

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
        return

    mapper = MapperAlgorithm(cover=cover, clustering=clustering)
    X = df.to_numpy()
    y = lens(X)
    df_y = pd.DataFrame(y, columns=[f"{lens_type} {i}" for i in range(y.shape[1])])
    if cover_scale_data:
        y = StandardScaler().fit_transform(y)
    if clustering_scale_data:
        X = StandardScaler().fit_transform(X)
    mapper_graph = mapper.fit_transform(X, y)
    return mapper_graph, df_y


def create_mapper_figure(df_X, df_y, df_target, mapper_graph, **kwargs):
    df_colors = pd.concat([df_target, df_y, df_X], axis=1)
    mapper_config = MapperConfig(**kwargs)
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
    logger.info("Mapper run completed successfully.")
    return mapper_fig


class App:

    def __init__(self, storage):
        self.storage = storage

        ui.colors(
            themelight="#ebedf8",
            themedark="#132f48",
        )

        with ui.left_drawer(elevated=True).classes(
            "w-96 h-full overflow-y-auto gap-12"
        ):
            with ui.link(target=GIT_REPO_URL, new_tab=True).classes("w-full"):
                ui.image(LOGO_URL)

            with ui.column().classes("w-full gap-2"):
                self._init_about()

            with ui.column().classes("w-full gap-2"):
                self._init_file_upload()

            ui.button(
                "⬆️ Load Data",
                on_click=self.load_file,
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

    def _init_about(self):
        with ui.dialog() as dialog, ui.card():
            ui.markdown(
                """
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
            )
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

    def _init_file_upload(self):
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

    def _init_lens(self):
        ui.label("🔎 Lens").classes("text-h6")
        self._init_lens_settings()

    def _init_lens_settings(self):
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

    def _init_cover(self):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("🌐 Cover").classes("text-h6")
            self.cover_scale = ui.switch(
                text="Scaling",
                value=COVER_SCALE_DATA,
            )
        self._init_cover_settings()

    def _init_cover_settings(self):
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

    def _init_clustering(self):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("🧮 Clustering").classes("text-h6")
            self.clustering_scale = ui.switch(
                text="Scaling",
                value=CLUSTERING_SCALE_DATA,
            )
        self._init_clustering_settings()

    def _init_clustering_settings(self):
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

    def _init_draw(self):
        ui.label("🎨 Draw").classes("text-h6")
        self._init_draw_settings()

    def _init_draw_settings(self):
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

    def _init_footnotes(self):
        ui.label(text=("Made in Rome, with ❤️ and ☕️.")).classes(
            "text-caption text-gray-500"
        ).classes("text-caption text-gray-500")

    def _init_draw_area(self):
        self.plot_container = ui.element("div").classes("w-full h-full")
        self.draw_area = None

    def get_mapper_config(self):
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
            plot_dimensions=(
                int(self.plot_dimensions.value)
                if self.plot_dimensions.value
                else PLOT_DIMENSIONS
            ),
            plot_iterations=(
                int(self.plot_iterations.value)
                if self.plot_iterations.value
                else PLOT_ITERATIONS
            ),
            plot_seed=(
                int(self.plot_seed.value) if self.plot_seed.value else RANDOM_SEED
            ),
        )

    def upload_file(self, file):
        if file is not None:
            df = pd.read_csv(file.content)
            self.storage["df"] = fix_data(df)
            self.storage["labels"] = pd.DataFrame()
            logger.info("File uploaded successfully.")
            ui.notify("File uploaded successfully.", type="info")
        else:
            logger.info("No file uploaded.")

    def load_file(self):
        if self.load_type.value == LOAD_EXAMPLE:
            if self.load_example.value == LOAD_EXAMPLE_DIGITS:
                df, labels = load_digits(as_frame=True, return_X_y=True)
            elif self.load_example.value == LOAD_EXAMPLE_IRIS:
                df, labels = load_iris(as_frame=True, return_X_y=True)
            else:
                logger.error("Unknown example dataset selected.")
                return
            self.storage["df"] = fix_data(df)
            self.storage["labels"] = fix_data(labels)
        elif self.load_type.value == LOAD_CSV:
            df = self.storage.get("df")
            if df is None:
                logger.warning("No data found. Please upload a file first.")
                ui.notify("No data found. Please upload a file first.", type="warning")
                return
        else:
            logger.error("Unknown load type selected.")
            return

        df = self.storage.get("df")
        if df is not None:
            logger.info("Data loaded successfully.")
            ui.notify("File loaded successfully.", type="positive")
        else:
            logger.warning("No data found. Please upload a file first.")
            ui.notify("No data found. Please upload a file first.", type="warning")

    async def async_run_mapper(self):
        notification = ui.notification(timeout=None, type="ongoing")
        notification.message = "Running Mapper..."
        notification.spinner = True
        df_X = self.storage.get("df")
        if df_X is None or df_X.empty:
            logger.warning("No data found. Please upload a file first.")
            ui.notify("No data found. Please upload a file first.", type="warning")
            return
        mapper_config = self.get_mapper_config()
        mapper_graph, df_y = await run.cpu_bound(
            run_mapper, df_X, **asdict(mapper_config)
        )
        self.storage["mapper_graph"] = mapper_graph
        self.storage["df_y"] = df_y
        notification.message = "Done!"
        notification.spinner = False
        notification.dismiss()

        await self.async_draw_mapper()

    async def async_draw_mapper(self):
        notification = ui.notification(timeout=None, type="ongoing")
        notification.message = "Drawing Mapper..."
        notification.spinner = True

        mapper_config = self.get_mapper_config()

        df_X = self.storage.get("df", pd.DataFrame())
        df_y = self.storage.get("df_y", pd.DataFrame())
        df_target = self.storage.get("labels", pd.DataFrame())
        mapper_graph = self.storage.get("mapper_graph", None)

        if df_X.empty or mapper_graph is None:
            logger.warning("No data or Mapper graph found. Please run Mapper first.")
            ui.notify(
                "No data or Mapper graph found. Please run Mapper first.",
                type="warning",
            )
            notification.message = "No data or Mapper graph found."
            notification.spinner = False
            notification.dismiss()
            return

        mapper_fig = await run.cpu_bound(
            create_mapper_figure,
            df_X,
            df_y,
            df_target,
            mapper_graph,
            **asdict(mapper_config),
        )

        logger.info("Displaying Mapper plot.")
        if self.draw_area is not None:
            self.draw_area.clear()
            self.plot_container.clear()
        with self.plot_container:
            self.draw_area = ui.plotly(mapper_fig).classes("w-full h-full")

        notification.message = "Done!"
        notification.spinner = False
        notification.dismiss()


@ui.page("/")
def main_page():
    ui.query(".nicegui-content").classes("p-0")
    storage = app.storage.client
    App(storage=storage)


def main():
    port = os.getenv("PORT", "8080")
    host = os.getenv("HOST", "0.0.0.0")
    production = os.getenv("PRODUCTION", "false").lower() == "true"
    storage_secret = os.getenv("STORAGE_SECRET", "storage_secret")
    ui.run(
        storage_secret=storage_secret,
        reload=not production,
        host=host,
        title="tda-mapper-app",
        favicon=ICON_URL,
        port=int(port),
    )
