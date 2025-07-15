"""
Test utilities.
"""

import numpy as np


def dataset_simple():
    """
    Create a simple dataset of points in a 2D space.

    This dataset consists of four points forming the corners of a rectangle
    such that two sides are longer than the other two.
    """
    return np.array(
        [
            [0.0, 1.0],
            [1.1, 0.0],
            [0.0, 0.0],
            [1.1, 1.0],
        ]
    )


def dataset_random(dim=1, num=1000):
    """
    Create a random dataset of points in the unit square.
    """
    return np.array([np.random.rand(dim) for _ in range(num)])


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
