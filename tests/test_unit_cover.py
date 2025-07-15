"""
Unit tests for the cover algorithms.
"""

import numpy as np
import pytest

from tdamapper.core import TrivialCover
from tdamapper.cover import (
    BallCover,
    CubicalCover,
    KNNCover,
    ProximityCubicalCover,
    StandardCubicalCover,
)
from tdamapper.utils.unionfind import UnionFind


def dataset_simple():
    """
    Create a simple dataset of points in a 2D space.

    This dataset consists of four points forming the corners of a rectangle
    such that two sides are longer than the other two.
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


SIMPLE = dataset_simple()

TWO_LINES = dataset_two_lines()

GRID = dataset_grid(10)


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

    uf = UnionFind(list(range(len(charts))))
    for point_id, chart_ids in point_charts.items():
        for i in range(len(chart_ids) - 1):
            uf.union(chart_ids[i], chart_ids[i + 1])
    # Count the number of unique components
    unique_components = set()
    for chart_id in range(len(charts)):
        unique_components.add(uf.find(chart_id))
    return len(unique_components)


@pytest.mark.parametrize(
    "dataset, cover, num_charts, num_components",
    [
        # Simple dataset
        (SIMPLE, TrivialCover(), 1, 1),
        # BallCover: components are expected to merge when the radius crosses the
        # lenghts of the rectangle sides.
        (SIMPLE, BallCover(radius=0.9, metric="euclidean"), 4, 4),
        (SIMPLE, BallCover(radius=1.1, metric="euclidean"), 2, 2),
        (SIMPLE, BallCover(radius=1.5, metric="euclidean"), 1, 1),
        # KNNCover: components are expected to merge when the number of neighbors
        # is enough to cover a given number of rectangle sides.
        (SIMPLE, KNNCover(neighbors=1, metric="euclidean"), 4, 4),
        (SIMPLE, KNNCover(neighbors=2, metric="euclidean"), 2, 2),
        (SIMPLE, KNNCover(neighbors=3, metric="euclidean"), 2, 1),
        # StandardCubicalCover: components are expected to merge when intervals
        # are big enough to cover the rectangle sides.
        (SIMPLE, StandardCubicalCover(n_intervals=2, overlap_frac=0.1), 4, 4),
        (SIMPLE, StandardCubicalCover(n_intervals=2, overlap_frac=0.5), 4, 4),
        (SIMPLE, StandardCubicalCover(n_intervals=1, overlap_frac=0.5), 1, 1),
        (SIMPLE, ProximityCubicalCover(n_intervals=2, overlap_frac=0.1), 4, 4),
        (SIMPLE, ProximityCubicalCover(n_intervals=2, overlap_frac=0.5), 4, 4),
        (SIMPLE, ProximityCubicalCover(n_intervals=1, overlap_frac=0.5), 1, 1),
        # Two lines dataset
        (TWO_LINES, TrivialCover(), 1, 1),
        # BallCover: components are expected to merge when the radius crosses the
        # distance between the two lines.
        (TWO_LINES, BallCover(radius=0.2, metric="euclidean"), 10, 2),
        (TWO_LINES, BallCover(radius=0.5, metric="euclidean"), 4, 2),
        (TWO_LINES, BallCover(radius=1.0, metric="euclidean"), 4, 2),
        (TWO_LINES, BallCover(radius=1.1, metric="euclidean"), 2, 1),
        (TWO_LINES, BallCover(radius=1.5, metric="euclidean"), 1, 1),
        # KNNCover: components are expected to merge when the number of neighbors
        # is more than the cardinality of a single line.
        (TWO_LINES, KNNCover(neighbors=3, metric="euclidean"), None, 2),
        (TWO_LINES, KNNCover(neighbors=10, metric="euclidean"), None, 2),
        (TWO_LINES, KNNCover(neighbors=100, metric="euclidean"), None, 2),
        (TWO_LINES, KNNCover(neighbors=1001, metric="euclidean"), 2, 1),
        # StandardCubicalCover: components are expected to merge when intervals
        # are big enough to cover the distance between the two lines.
        (TWO_LINES, StandardCubicalCover(n_intervals=2, overlap_frac=0.5), 4, 2),
        (TWO_LINES, ProximityCubicalCover(n_intervals=2, overlap_frac=0.5), 4, 2),
        # Grid dataset
        (GRID, TrivialCover(), 1, 1),
        # BallCover: components are expected to jump from many singletons sets
        # to a single one when the radius crosses the grid spacing.
        (GRID, BallCover(radius=0.01, metric="euclidean"), 100, 100),
        (GRID, BallCover(radius=0.2, metric="euclidean"), None, 1),
        # KNNCover: components are expected to merge when the number of neighbors
        # is more than the number of adjacent points in the grid.
        (GRID, KNNCover(neighbors=1, metric="euclidean"), 100, 100),
        (GRID, KNNCover(neighbors=10, metric="euclidean"), None, 1),
        (GRID, StandardCubicalCover(n_intervals=2, overlap_frac=0.5), 4, 1),
        (GRID, ProximityCubicalCover(n_intervals=2, overlap_frac=0.5), 4, 1),
    ],
)
def test_cover(dataset, cover, num_charts, num_components):
    """
    Test that the cover algorithm covers the dataset correctly, and that the
    number of charts and components is as expected. If num_charts or
    num_components is None, the test will not check that value.
    """
    charts = assert_coverage(dataset, cover)
    if num_charts is not None:
        assert len(charts) == num_charts
    if num_components is not None:
        assert count_components(charts) == num_components


@pytest.mark.parametrize(
    "cover, params",
    [
        (TrivialCover(), {}),
        (
            BallCover(radius=0.2, metric="euclidean"),
            {"radius": 0.21, "metric": "euclidean"},
        ),
        (
            KNNCover(neighbors=10, metric="euclidean"),
            {"neighbors": 13, "metric": "euclidean"},
        ),
        (
            StandardCubicalCover(n_intervals=2, overlap_frac=0.5),
            {"n_intervals": 4, "overlap_frac": 0.145},
        ),
        (
            ProximityCubicalCover(n_intervals=2, overlap_frac=0.5),
            {"n_intervals": 4, "overlap_frac": 0.145},
        ),
        (
            CubicalCover(n_intervals=2, overlap_frac=0.5, algorithm="standard"),
            {"n_intervals": 4, "overlap_frac": 0.145, "algorithm": "proximity"},
        ),
    ],
)
def test_params(cover, params):
    """
    Test that the cover can get and set parameters correctly.
    """
    cover.set_params(**params)
    params = cover.get_params(deep=True)
    for k, v in params.items():
        assert params[k] == v
