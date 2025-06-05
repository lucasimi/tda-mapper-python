import logging
from dataclasses import asdict, dataclass

import pandas as pd
from nicegui import app, run, ui
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from umap import UMAP

from tdamapper.clustering import TrivialClustering
from tdamapper.cover import BallCover, CubicalCover, KNNCover
from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LENS_IDENTITY = "Identity"
LENS_PCA = "PCA"
LENS_UMAP = "UMAP"

LENS_PCA_N_COMPONENTS = 2
LENS_UMAP_N_COMPONENTS = 2

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

CLUSTERING_KMEANS_N_CLUSTERS = 2
CLUSTERING_DBSCAN_EPS = 0.5
CLUSTERING_DBSCAN_MIN_SAMPLES = 5
CLUSTERING_AGGLOMERATIVE_N_CLUSTERS = 2

RANDOM_SEED = 42


@dataclass
class MapperConfig:
    lens_type: str = LENS_PCA
    cover_type: str = COVER_CUBICAL
    clustering_type: str = CLUSTERING_TRIVIAL
    lens_pca_n_components: int = LENS_PCA_N_COMPONENTS
    lens_umap_n_components: int = LENS_UMAP_N_COMPONENTS
    cover_cubical_n_intervals: int = COVER_CUBICAL_N_INTERVALS
    cover_cubical_overlap_frac: float = COVER_CUBICAL_OVERLAP_FRAC
    cover_knn_neighbors: int = COVER_KNN_NEIGHBORS
    clustering_kmeans_n_clusters: int = CLUSTERING_KMEANS_N_CLUSTERS
    clustering_dbscan_eps: float = CLUSTERING_DBSCAN_EPS
    clustering_dbscan_min_samples: int = CLUSTERING_DBSCAN_MIN_SAMPLES
    clustering_agglomerative_n_clusters: int = CLUSTERING_AGGLOMERATIVE_N_CLUSTERS


def identity(X):
    return X


def pca(n_components):

    def _pca(X):
        pca_model = PCA(n_components=n_components, random_state=RANDOM_SEED)
        return pca_model.fit_transform(X)

    return _pca


def umap(n_components):

    def _umap(X):
        umap_model = umap.UMAP(n_components=n_components, random_state=RANDOM_SEED)
        return umap_model.fit_transform(X)

    return _umap


def run_mapper(df, **kwargs):
    if df is None:
        logger.error("No data found. Please upload a file first.")
        return
    logger.info("Computing Mapper.")

    mapper_config = MapperConfig(**kwargs)

    lens_type = mapper_config.lens_type
    cover_type = mapper_config.cover_type
    clustering_type = mapper_config.clustering_type
    lens_pca_n_components = mapper_config.lens_pca_n_components
    lens_umap_n_components = mapper_config.lens_umap_n_components
    cover_cubical_n_intervals = mapper_config.cover_cubical_n_intervals
    cover_cubical_overlap_frac = mapper_config.cover_cubical_overlap_frac
    cover_knn_neighbors = mapper_config.cover_knn_neighbors
    clustering_kmeans_n_clusters = mapper_config.clustering_kmeans_n_clusters
    clustering_dbscan_eps = mapper_config.clustering_dbscan_eps
    clustering_dbscan_min_samples = mapper_config.clustering_dbscan_min_samples
    clustering_agglomerative_n_clusters = (
        mapper_config.clustering_agglomerative_n_clusters
    )

    if lens_type == LENS_IDENTITY:
        lens = identity
    elif lens_type == LENS_PCA:
        lens = pca(n_components=lens_pca_n_components)
    elif lens_type == LENS_UMAP:
        lens = umap(n_components=lens_umap_n_components)

    if cover_type == COVER_CUBICAL:
        cover = CubicalCover(n_intervals=cover_cubical_n_intervals)
    elif cover_type == COVER_BALL:
        cover = BallCover(overlap_fraction=cover_cubical_overlap_frac)
    elif cover_type == COVER_KNN:
        cover = KNNCover(n_neighbors=cover_knn_neighbors)
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
            random_state=RANDOM_SEED,
        )
    elif clustering_type == CLUSTERING_AGGLOMERATIVE:
        clustering = AgglomerativeClustering(
            n_clusters=clustering_agglomerative_n_clusters,
            random_state=RANDOM_SEED,
        )
    else:
        logger.error(f"Unknown clustering type: {clustering_type}")
        return

    mapper = MapperAlgorithm(cover=cover, clustering=clustering)
    X = df.to_numpy()
    y = lens(X)
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
        ui.query("body").style("margin: 0; padding: 0; overflow: hidden;")
        with ui.row().classes("w-full h-screen m-0 p-0 gap-0 overflow-hidden"):
            with ui.column().classes("w-64 h-full m-0 p-0 overflow-y-auto"):
                self._init_file_upload()
                self._init_lens()
                self._init_cover()
                self._init_clustering()

                ui.button(
                    "Run Mapper",
                    on_click=self.async_run_mapper,
                    color="primary",
                ).classes("w-full")

            with ui.column().classes("flex-grow h-full overflow-hidden p-0 m-0"):
                self._init_plot()

    def _init_file_upload(self):
        with ui.card().tight().classes("w-full p-3"):
            ui.upload(
                on_upload=self.upload_file,
                auto_upload=True,
                label="Upload CSV File",
            ).classes("w-full")
            with ui.card_section().classes("w-full"):
                ui.button("Load", on_click=self.load_file).classes("w-full")

    def _init_lens(self):
        with ui.card().classes("w-full p-3"):
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
        with ui.card().classes("w-full p-3"):
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
                label="Ball Radius",
                value=COVER_BALL_RADIUS,
            ).classes("w-full")
            self.cover_cubical_overlap_frac.bind_visibility_from(
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
        with ui.card().classes("w-full p-3"):
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
        self.plot_container = ui.card().classes("w-full h-full m-0 p-0 overflow-hidden")

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
            logger.info(f"{df.head()}")
        else:
            logger.info("No file uploaded.")

    def load_file(self):
        df = self.storage.get("df")
        if df is not None:
            logger.info("Data loaded successfully.")
        else:
            logger.warning("No data found. Please upload a file first.")

    async def async_run_mapper(self):
        df = self.storage.get("df")
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
    storage = app.storage.client
    App(storage=storage)


ui.run(storage_secret="secret")
