# tda-mapper

[![test](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml)
[![deploy](https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml/badge.svg)](https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml)
[![docs](https://readthedocs.org/projects/tda-mapper/badge/?version=latest)](https://tda-mapper.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R)](https://codecov.io/github/lucasimi/tda-mapper-python) 

The Mapper algorithm is a well-known technique in the field of topological data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining, and social sciences, due to its ability to preserve topological features of the underlying space, providing a visual representation that facilitates exploration and interpretation.
For an in-depth coverage of Mapper you can read [the original paper](https://research.math.osu.edu/tgda/mapperPBG.pdf). 

This Python package provides a simple and efficient implementation of the Mapper algorithm.

* Installation from package: ```python -m pip install tda-mapper```

* Installation from sources: clone this repo and run ```python -m pip install .```

* Documentation: https://tda-mapper.readthedocs.io/en/latest/ 

## Usage

[Here](https://github.com/lucasimi/tda-mapper-python/blob/7cd3814034eea4cb7be8917bfc9b5ee2357ed8d1/tests/example.py) you can find a worked out example that shows how to use this package. 
In the example we perform some analysis on the the well known dataset of [hand written digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

https://github.com/lucasimi/tda-mapper-python/blob/7cd3814034eea4cb7be8917bfc9b5ee2357ed8d1/tests/example.py


![Mapper graph of digits, colored according to mean](https://github.com/lucasimi/tda-mapper-python/blob/7cd3814034eea4cb7be8917bfc9b5ee2357ed8d1/resources/digits_mean.png)

It's also possible to obtain a new plot colored according to different values, while keeping the same computed geometry. For example, if we want to visualize how much dispersion we have on each cluster, we could plot colors according to the standard deviation.

![Mapper graph of digits, colored according to std](https://github.com/lucasimi/tda-mapper-python/blob/7cd3814034eea4cb7be8917bfc9b5ee2357ed8d1/resources/digits_std.png)

The mapper graph of the digits dataset shows a few interesting patterns. For example, we can make the following observations:

* Clusters that share the same color are all connected together, and located in the same area of the graph. 
This behavior is present in those digits which are easy to tell apart from the others, for example digits 0 and 4.

* Some clusters are not well separated and tend to overlap one on the other. 
This mixed behavior is present in those digits which can be easily confused one with the other, for example digits 5 and 6.

* Clusters located across the "boundary" of two different digits show a transition either due to a change in distribution or due to distorsions in the hand written text, for example digits 8 and 2.
