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
        node_size=[0.0, 0.5, 1.0],
    )
    logger.info("Mapper run completed successfully.")
    return mapper_fig


@ui.page("/")
def index():
    storage = app.storage.client

    def upload_file(file):
        if file is not None:
            df = pd.read_csv(file.content)
            storage["df"] = df

            logger.info("File uploaded successfully.")
            logger.info(f"{df.head()}")
        else:
            logger.info("No file uploaded.")

    def load_file():
        df = storage.get("df")
        if df is not None:
            logger.info("Data loaded successfully.")
        else:
            logger.warning("No data found. Please upload a file first.")

    def get_mapper_config():
        return MapperConfig(
            lens_type=str(lens.value) if lens.value else LENS_PCA,
            cover_type=str(cover.value) if cover.value else COVER_CUBICAL,
            clustering_type=(
                str(clustering.value) if clustering.value else CLUSTERING_TRIVIAL
            ),
            lens_pca_n_components=(
                int(pca_n_components.value)
                if pca_n_components.value
                else LENS_PCA_N_COMPONENTS
            ),
            lens_umap_n_components=(
                int(umap_n_components.value)
                if umap_n_components.value
                else LENS_UMAP_N_COMPONENTS
            ),
            cover_cubical_n_intervals=(
                int(n_intervals.value)
                if n_intervals.value
                else COVER_CUBICAL_N_INTERVALS
            ),
            cover_cubical_overlap_frac=(
                float(overlap_frac.value)
                if overlap_frac.value
                else COVER_CUBICAL_OVERLAP_FRAC
            ),
            cover_knn_neighbors=(
                int(neighbors.value) if neighbors.value else COVER_KNN_NEIGHBORS
            ),
            clustering_kmeans_n_clusters=(
                int(kmeans_n_clusters.value)
                if kmeans_n_clusters.value
                else CLUSTERING_KMEANS_N_CLUSTERS
            ),
            clustering_dbscan_eps=(
                float(dbscan_eps.value) if dbscan_eps.value else CLUSTERING_DBSCAN_EPS
            ),
            clustering_dbscan_min_samples=(
                int(dbscan_min_samples.value)
                if dbscan_min_samples.value
                else CLUSTERING_DBSCAN_MIN_SAMPLES
            ),
            clustering_agglomerative_n_clusters=(
                int(agglomerative_n_clusters.value)
                if agglomerative_n_clusters.value
                else CLUSTERING_AGGLOMERATIVE_N_CLUSTERS
            ),
        )

    async def async_run_mapper():
        df = storage.get("df")
        mapper_config = get_mapper_config()
        mapper_fig = await run.cpu_bound(run_mapper, df, **asdict(mapper_config))
        mapper_fig.layout.width = None
        mapper_fig.layout.autosize = True
        plot_container.clear()
        with plot_container:
            logger.info("Displaying Mapper plot.")
            ui.plotly(mapper_fig)

    with ui.row().classes("w-full h-screen m-0 p-0 gap-0 overflow-hidden"):

        with ui.column().classes("w-64 h-full m-0 p-0"):
            with ui.card().tight().classes("w-full"):
                ui.upload(
                    on_upload=upload_file,
                    auto_upload=True,
                    label="Upload CSV File",
                ).classes("w-full")
                with ui.card_section().classes("w-full"):
                    ui.button("Load", on_click=load_file).classes("w-full")

            with ui.card().classes("w-full"):
                lens = ui.select(
                    options=[
                        LENS_IDENTITY,
                        LENS_PCA,
                        LENS_UMAP,
                    ],
                    label="Lens",
                    value=LENS_PCA,
                ).classes("w-full")

                pca_n_components = ui.number(
                    label="PCA Components",
                    value=LENS_PCA_N_COMPONENTS,
                ).classes("w-full")
                pca_n_components.bind_visibility_from(
                    target_object=lens,
                    target_name="value",
                    value=LENS_PCA,
                )

                umap_n_components = ui.number(
                    label="UMAP Components",
                    value=LENS_UMAP_N_COMPONENTS,
                ).classes("w-full")
                umap_n_components.bind_visibility_from(
                    target_object=lens,
                    target_name="value",
                    value=LENS_UMAP,
                )

            with ui.card().classes("w-full"):
                cover = ui.select(
                    options=[
                        COVER_CUBICAL,
                        COVER_BALL,
                        COVER_KNN,
                    ],
                    label="Cover",
                    value=COVER_CUBICAL,
                ).classes("w-full")

                n_intervals = ui.number(
                    label="Number of Intervals",
                    value=COVER_CUBICAL_N_INTERVALS,
                ).classes("w-full")
                n_intervals.bind_visibility_from(
                    target_object=cover,
                    target_name="value",
                    value=COVER_CUBICAL,
                )

                overlap_frac = ui.number(
                    label="Ball Radius",
                    value=COVER_BALL_RADIUS,
                ).classes("w-full")
                overlap_frac.bind_visibility_from(
                    target_object=cover,
                    target_name="value",
                    value=COVER_BALL,
                )

                neighbors = ui.number(
                    label="Number of Neighbors",
                    value=COVER_KNN_NEIGHBORS,
                ).classes("w-full")
                neighbors.bind_visibility_from(
                    target_object=cover,
                    target_name="value",
                    value=COVER_KNN,
                )

            with ui.card().classes("w-full"):
                clustering = ui.select(
                    options=[
                        CLUSTERING_TRIVIAL,
                        CLUSTERING_KMEANS,
                        CLUSTERING_DBSCAN,
                        CLUSTERING_AGGLOMERATIVE,
                    ],
                    label="Clustering",
                    value=CLUSTERING_TRIVIAL,
                ).classes("w-full")

                kmeans_n_clusters = ui.number(
                    label="Number of Clusters",
                    value=CLUSTERING_KMEANS_N_CLUSTERS,
                ).classes("w-full")
                kmeans_n_clusters.bind_visibility_from(
                    target_object=clustering,
                    target_name="value",
                    value=CLUSTERING_KMEANS,
                )

                dbscan_eps = ui.number(
                    label="Epsilon",
                    value=CLUSTERING_DBSCAN_EPS,
                ).classes("w-full")
                dbscan_eps.bind_visibility_from(
                    target_object=clustering,
                    target_name="value",
                    value=CLUSTERING_DBSCAN,
                )
                dbscan_min_samples = ui.number(
                    label="Min Samples",
                    value=CLUSTERING_DBSCAN_MIN_SAMPLES,
                ).classes("w-full")
                dbscan_min_samples.bind_visibility_from(
                    target_object=clustering,
                    target_name="value",
                    value=CLUSTERING_DBSCAN,
                )

                agglomerative_n_clusters = ui.number(
                    label="Number of Clusters",
                    value=CLUSTERING_AGGLOMERATIVE_N_CLUSTERS,
                ).classes("w-full")
                agglomerative_n_clusters.bind_visibility_from(
                    target_object=clustering,
                    target_name="value",
                    value=CLUSTERING_AGGLOMERATIVE,
                )

            ui.button(
                "Run Mapper",
                on_click=async_run_mapper,
            ).classes("w-full")

        with ui.column().classes("flex-1 h-full overflow-hidden m-0 p-0"):
            plot_container = ui.element("div").classes("w-full h-full")


ui.run(storage_secret="secret")
