import logging
from dataclasses import asdict, dataclass

import pandas as pd
from nicegui import app, run, ui
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
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
    if df is None:
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
    df_fixed = fix_data(df)
    X = df_fixed.to_numpy()
    y = lens(X)
    if cover_scale_data:
        y = StandardScaler().fit_transform(y)
    if clustering_scale_data:
        X = StandardScaler().fit_transform(X)
    mapper_graph = mapper.fit_transform(X, y)
    mapper_fig = MapperPlot(
        mapper_graph,
        dim=3,
    ).plot_plotly(
        colors=X,
        height=800,
        node_size=[i * 0.125 for i in range(17)],
    )
    logger.info("Mapper run completed successfully.")
    return mapper_fig


class App:

    def __init__(self, storage):
        self.storage = storage
        with ui.row().classes("w-full h-screen overflow-hidden p-0 m-0"):
            with ui.column().classes("w-96 h-full p-1 m-0"):
                with ui.link(target=GIT_REPO_URL, new_tab=True).classes("w-full"):
                    ui.image(LOGO_URL)
                with ui.column().classes("w-96 h-full overflow-y-auto p-1 m-0"):
                    with ui.card().classes("w-full"):
                        self._init_file_upload()
                    with ui.card().classes("w-full"):
                        self._init_lens()
                    with ui.card().classes("w-full"):
                        self._init_cover()
                    with ui.card().classes("w-full"):
                        self._init_clustering()

                    ui.button(
                        "Run Mapper",
                        on_click=self.async_run_mapper,
                        color="primary",
                    ).classes("w-full")
            with ui.column().classes("flex-1 h-full overflow-hidden p-1 m-0"):
                self._init_plot()

    def _init_file_upload(self):
        ui.upload(
            on_upload=self.upload_file,
            auto_upload=True,
            label="Upload CSV File",
        ).classes("w-full")
        with ui.card_section().classes("w-full"):
            ui.button("Load", on_click=self.load_file).classes("w-full")

    def _init_lens(self):
        ui.markdown("#### Lens")

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
        ui.markdown("#### Cover")

        self.cover_scale = ui.switch(
            text="Scale Data",
            value=COVER_SCALE_DATA,
        ).classes("w-full")

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
        ui.markdown("##### Clustering")

        self.clustering_scale = ui.switch(
            text="Scale Data",
            value=CLUSTERING_SCALE_DATA,
        ).classes("w-full")

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

    def _init_plot(self):
        self.plot_container = ui.card().classes("w-full h-full")

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
        )

    def upload_file(self, file):
        if file is not None:
            df = pd.read_csv(file.content)
            self.storage["df"] = df
            logger.info("File uploaded successfully.")
            ui.notify("File uploaded successfully.", type="info")
        else:
            logger.info("No file uploaded.")

    def load_file(self):
        df = self.storage.get("df")
        if df is not None:
            logger.info("Data loaded successfully.")
            ui.notify("File loaded successfully.", type="positive")
        else:
            logger.warning("No data found. Please upload a file first.")
            ui.notify("No data found. Please upload a file first.", type="warning")

    async def async_run_mapper(self):
        df = self.storage.get("df")
        if df is None or df.empty:
            logger.warning("No data found. Please upload a file first.")
            ui.notify("No data found. Please upload a file first.", type="warning")
            return
        ui.notify("Running Mapper...", type="info")
        with self.plot_container:
            ui.spinner(size="lg")
        mapper_config = self.get_mapper_config()
        mapper_fig = await run.cpu_bound(run_mapper, df, **asdict(mapper_config))
        mapper_fig.layout.width = None
        mapper_fig.layout.height = None
        mapper_fig.layout.autosize = True
        self.plot_container.clear()
        with self.plot_container:
            logger.info("Displaying Mapper plot.")
            ui.plotly(mapper_fig).classes("w-full h-full")


@ui.page("/")
def main():
    ui.query(".nicegui-content").classes("p-0")
    storage = app.storage.client
    App(storage=storage)


ui.run(storage_secret="secret")
