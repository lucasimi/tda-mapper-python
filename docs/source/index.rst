.. tda-mapper documentation master file, created by
   sphinx-quickstart on Fri Jan 26 21:56:08 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

|PyPI version| |downloads| |test| |deploy| |docs| |codecov| |DOI|

tda-mapper
==========

A simple and efficient implementation of Mapper algorithm for
Topological Data Analysis.

-  **Installation**: ``pip install tda-mapper``

-  **Documentation**: https://tda-mapper.readthedocs.io/en/main/

The Mapper algorithm is a well-known technique in the field of
topological data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining,
and social sciences, due to its ability to preserve topological features
of the underlying space, providing a visual representation that
facilitates exploration and interpretation. For an in-depth coverage of
Mapper you can read `the original
paper <https://research.math.osu.edu/tgda/mapperPBG.pdf>`__.

+-----------------+-----------------+-----------------+-----------------+
| Step 1          | Step 2          | Step 3          | Step 4          |
+=================+=================+=================+=================+
| |Step 1|        | |Step 2|        | |Step 3|        | |image1|        |
+-----------------+-----------------+-----------------+-----------------+

Examples
========

.. toctree::
   :maxdepth: 1

   notebooks/circles_online
   notebooks/digits_online

API Reference
=============
.. toctree::
   :maxdepth: 1

   tdamapper

Citations
=========

To cite **tda-mapper** in your work you can use the Zenodo archive
```https://doi.org/10.5281/zenodo.10642381`` <https://doi.org/10.5281/zenodo.10642381>`__.

In the archive you can find a permanent reference to the exact version
you used in your work.

For example, to cite version v0.4.0 you can use:

::

   Simi, L. (2024). tda-mapper (v0.4.0). Zenodo. https://doi.org/10.5281/zenodo.10655755

BibTeX entry:

::

   @software{tda-mapper_v0.4.0,
     author       = {Simi, Luca},
     title        = {tda-mapper},
     month        = feb,
     year         = 2024,
     publisher    = {Zenodo},
     version      = {v0.4.0},
     doi          = {10.5281/zenodo.10655755},
     url          = {https://doi.org/10.5281/zenodo.10655755}
   }

.. |PyPI version| image:: https://badge.fury.io/py/tda-mapper.svg
   :target: https://badge.fury.io/py/tda-mapper
.. |downloads| image:: https://img.shields.io/pypi/dm/tda-mapper
   :target: https://pypi.python.org/pypi/tda-mapper/
.. |test| image:: https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml
.. |deploy| image:: https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml/badge.svg
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/deploy.yml
.. |docs| image:: https://readthedocs.org/projects/tda-mapper/badge/?version=main
   :target: https://tda-mapper.readthedocs.io/en/main/?badge=main
.. |codecov| image:: https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R
   :target: https://codecov.io/github/lucasimi/tda-mapper-python
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10642381.svg
   :target: https://doi.org/10.5281/zenodo.10642381
.. |Step 1| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_1.png
.. |Step 2| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_2.png
.. |Step 3| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_3.png
.. |image1| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_4.png
.. |Dataset| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_dataset.png
.. |Mapper graph (average)| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_mean.png
.. |Mapper graph (standard deviation)| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_std.png
