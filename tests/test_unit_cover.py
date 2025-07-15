import numpy as np
import pytest

from tdamapper.core import TrivialCover
from tdamapper.cover import BallCover, CubicalCover, KNNCover


def dataset_simple():
    """
    Create a simple dataset of points in a 2D space.
    """
    return [
        np.array([0.0, 1.0]),
        np.array([1.1, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.1, 1.0]),
    ]


def dataset_random(dim=1, num=1000):
    """
    Create a random dataset of points in the unit square.
    """
    return [np.random.rand(dim) for _ in range(num)]


def dataset_two_lines(num=1000):
    """
    Create a dataset consisting of two lines in the unit square.
    One line is horizontal at y=0, the other is vertical at x=1.
    """
    t = np.linspace(0.0, 1.0, num)
    line1 = np.array([[x, 0.0] for x in t])
    line2 = np.array([[x, 1.0] for x in t])
    return np.concatenate((line1, line2), axis=0)


def dataset_grid(num=1000):
    """
    Create a grid dataset in the unit square.
    The grid consists of points evenly spaced in both dimensions.
    """
    t = np.linspace(0.0, 1.0, num)
    s = np.linspace(0.0, 1.0, num)
    grid = np.array([[x, y] for x in t for y in s])
    return grid


def assert_coverage(data, cover):
    """
    Assert that the cover applies to the data and covers all points.
    """
    covered = set()
    charts = list(cover.apply(data))
    for point_ids in charts:
        for point_id in point_ids:
            covered.add(point_id)
    assert len(covered) == len(data)
    return charts


def count_components(charts):
    """
    Count the number of unique connected components in the charts.
    Each chart is a list of point ids. When multiple charts share points,
    they are considered connected.
    """
    # Create a mapping from point ids to chart ids
    point_charts = {}
    for chart_id, point_ids in enumerate(charts):
        for point_id in point_ids:
            if point_id not in point_charts:
                point_charts[point_id] = []
            point_charts[point_id].append(chart_id)

    chart_components = {x: x for x in range(len(charts))}
    for point_id, chart_ids in point_charts.items():
        if len(chart_ids) > 1:
            # Union all chart ids for this point
            first_chart = chart_ids[0]
            for chart_id in chart_ids[1:]:
                chart_components[chart_id] = chart_components[first_chart]
    # Count unique components
    unique_components = set(chart_components.values())
    return len(unique_components)


def test_trivial_cover_random():
    data = dataset_random()
    cover = TrivialCover()
    assert_coverage(data, cover)


def test_trivial_cover_two_lines():
    data = dataset_two_lines()
    cover = TrivialCover()
    charts = assert_coverage(data, cover)
    assert 1 == len(charts)
    num_components = count_components(charts)
    assert 1 == num_components


@pytest.mark.parametrize(
    "dataset, cover, num_charts, num_components",
    [
        # Simple dataset tests
        (dataset_simple(), TrivialCover(), 1, 1),
        (dataset_simple(), BallCover(radius=1.1, metric="euclidean"), 2, 2),
        (dataset_simple(), KNNCover(neighbors=2, metric="euclidean"), 2, 2),
        (dataset_simple(), CubicalCover(n_intervals=2, overlap_frac=0.5), 4, None),
        # Two lines dataset tests
        (dataset_two_lines(), TrivialCover(), 1, 1),
        (dataset_two_lines(), BallCover(radius=0.2, metric="euclidean"), None, 2),
        (dataset_two_lines(), KNNCover(neighbors=10, metric="euclidean"), None, 2),
        (dataset_two_lines(), CubicalCover(n_intervals=2, overlap_frac=0.5), 4, None),
        # Grid dataset tests
        (dataset_grid(), TrivialCover(), 1, 1),
        (dataset_grid(), BallCover(radius=0.05, metric="euclidean"), None, 1),
        (dataset_grid(), KNNCover(neighbors=10, metric="euclidean"), None, 1),
        (dataset_grid(), CubicalCover(n_intervals=2, overlap_frac=0.5), 4, None),
    ],
)
def test_cover(dataset, cover, num_charts, num_components):
    charts = assert_coverage(dataset, cover)
    if num_charts is not None:
        assert len(charts) == num_charts
    if num_components is not None:
        assert count_components(charts) == num_components


def test_trivial_cover_grid():
    data = dataset_two_lines()
    cover = TrivialCover()
    charts = assert_coverage(data, cover)
    assert 1 == len(charts)
    num_components = count_components(charts)
    assert 1 == num_components


def test_ball_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    ]
    cover = BallCover(radius=1.1, metric="euclidean")
    charts = assert_coverage(data, cover)
    assert 2 == len(charts)
    num_components = count_components(charts)
    assert 1 == num_components


def test_ball_cover_random():
    data = dataset_random(dim=2, num=10)
    cover = BallCover(radius=0.2, metric="euclidean")
    assert_coverage(data, cover)


def test_ball_cover_two_lines():
    data = dataset_two_lines()
    cover = BallCover(radius=0.2, metric="euclidean")
    charts = assert_coverage(data, cover)
    num_components = count_components(charts)
    assert 2 == num_components


def test_ball_cover_grid():
    data = dataset_grid(num=100)
    cover = BallCover(radius=0.05, metric="euclidean")
    charts = assert_coverage(data, cover)
    num_components = count_components(charts)
    assert 1 == num_components


def test_knn_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.1, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.1, 1.0]),
    ]
    cover = KNNCover(neighbors=2, metric="euclidean")
    charts = assert_coverage(data, cover)
    assert 2 == len(charts)


def test_knn_cover_two_lines():
    data = dataset_two_lines()
    cover = KNNCover(neighbors=10, metric="euclidean")
    charts = assert_coverage(data, cover)
    num_components = count_components(charts)
    assert 2 == num_components


def test_knn_cover_grid():
    data = dataset_grid(num=100)
    cover = KNNCover(neighbors=10, metric="euclidean")
    charts = assert_coverage(data, cover)
    num_components = count_components(charts)
    assert 1 == num_components


def test_cubical_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.1, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.1, 1.0]),
    ]
    cover = CubicalCover(n_intervals=2, overlap_frac=0.5)
    charts = list(cover.apply(data))
    assert 4 == len(charts)


def test_cubical_cover_random():
    data = dataset_random(dim=2, num=100)
    cover = CubicalCover(n_intervals=5, overlap_frac=0.1)
    assert_coverage(data, cover)


def test_cubical_cover_params():
    cover = CubicalCover(n_intervals=2, overlap_frac=0.5)
    params = cover.get_params(deep=True)
    assert 2 == params["n_intervals"]
    assert 0.5 == params["overlap_frac"]


def test_standard_cover_simple():
    data = [
        np.array([0.0, 1.0]),
        np.array([1.1, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.1, 1.0]),
    ]
    cover = CubicalCover(
        n_intervals=2,
        overlap_frac=0.5,
        algorithm="standard",
    )
    charts = list(cover.apply(data))
    assert 4 == len(charts)
