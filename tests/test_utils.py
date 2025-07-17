"""
Test utilities.
"""

import random

import numpy as np


def list_int_random(num=1000):
    """
    Create a list of random integers.
    """
    return random.sample(list(range(num)), num)


def dataset_empty():
    """
    Create an empty dataset.
    """
    return np.array([])


def dataset_simple(scale=1.0):
    """
    Create a simple dataset of points in a 2D space.

    This dataset consists of four points forming the corners of a rectangle
    such that two sides are longer than the other two.
    """
    return scale * np.array(
        [
            [0.0, 1.0],
            [1.1, 0.0],
            [0.0, 0.0],
            [1.1, 1.0],
        ]
    )


def dataset_random(dim=1, num=1000, scale=1.0):
    """
    Create a random dataset of points in the unit square.
    """
    return np.array([scale * np.random.rand(dim) for _ in range(num)])


def dataset_two_lines(num=1000, scale=1.0):
    """
    Create a dataset consisting of two lines in the unit square.
    One line is horizontal at y=0, the other is vertical at x=1.
    """
    t = np.linspace(0.0, 1.0, num)
    line1 = scale * np.array([[x, 0.0] for x in t])
    line2 = scale * np.array([[x, 1.0] for x in t])
    return np.concatenate((line1, line2), axis=0)


def dataset_grid(num=1000, scale=1.0):
    """
    Create a grid dataset in the unit square.
    The grid consists of points evenly spaced in both dimensions.
    """
    t = np.linspace(0.0, 1.0, num)
    s = np.linspace(0.0, 1.0, num)
    grid = scale * np.array([[x, y] for x in t for y in s])
    return grid
