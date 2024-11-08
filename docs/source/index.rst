.. |PyPI version| image:: https://badge.fury.io/py/tda-mapper.svg
   :target: https://badge.fury.io/py/tda-mapper
.. |downloads| image:: https://img.shields.io/pypi/dm/tda-mapper
   :target: https://pypi.python.org/pypi/tda-mapper/
.. |test| image:: https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml/badge.svg
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml
.. |publish| image:: https://github.com/lucasimi/tda-mapper-python/actions/workflows/publish.yml/badge.svg
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/publish.yml
.. |docs| image:: https://readthedocs.org/projects/tda-mapper/badge/?version=main
   :target: https://tda-mapper.readthedocs.io/en/main/?badge=main
.. |codecov| image:: https://codecov.io/github/lucasimi/tda-mapper-python/graph/badge.svg?token=FWSD8JUG6R
   :target: https://codecov.io/github/lucasimi/tda-mapper-python
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10642381.svg
   :target: https://doi.org/10.5281/zenodo.10642381
.. |Streamlit App| image:: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
   :target: https://tda-mapper-app.streamlit.app/
.. |Step 1| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_1.png
.. |Step 2| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_2.png
.. |Step 3| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_3.png
.. |Step 4| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_4.png

.. figure::
   https://github.com/lucasimi/tda-mapper-python/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png
   :alt: Logo

|PyPI version| |downloads| |test| |publish| |docs| |codecov| |DOI|
|Streamlit App|

tda-mapper
==========

**tda-mapper** is a simple and efficient Python library implementing the
Mapper algorithm for Topological Data Analysis (TDA). It enables fast
computation of Mapper graphs using *vp-trees* to optimize the
construction of open covers for enhanced performance and scalability.

For further details, please refer to our
`preprint <https://doi.org/10.5281/zenodo.10659651>`__.

- **Installation**: ``pip install tda-mapper``

- **Documentation**:
  `Online on Read the Docs <https://tda-mapper.readthedocs.io/en/main/>`__.

- **Interactive App**:
  `Live Demo on Streamlit Cloud <https://tda-mapper-app.streamlit.app/>`__,
  or run Locally with:

  ::

      pip install -r app/requirements.txt
      streamlit run app/streamlit_app.py

Features
--------

- **Efficient Mapper Computation**: Optimized for higher-dimensional
  lenses.

- **Interactive Visualizations**: Multiple plotting backends for
  flexibility.

- **Interactive App**: Interactive tool for quick, in-depth data
  exploration.

Background
----------

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
| |Step 1|        | |Step 2|        | |Step 3|        | |Step 4|        |
+-----------------+-----------------+-----------------+-----------------+
| Chose lens      | Cover image     | Run clustering  | Build graph     |
+-----------------+-----------------+-----------------+-----------------+

.. toctree::
   :maxdepth: 1

   quickstart
   examples 
   citations
   apiref
