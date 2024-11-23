.. |PyPI version| image:: https://img.shields.io/pypi/v/tda-mapper?logo=python&logoColor=silver
   :target: https://pypi.python.org/pypi/tda-mapper
.. |downloads| image:: https://img.shields.io/pypi/dm/tda-mapper?logo=python&logoColor=silver
   :target: https://pypi.python.org/pypi/tda-mapper
.. |test| image:: https://img.shields.io/github/actions/workflow/status/lucasimi/tda-mapper-python/test.yml?logo=github&logoColor=silver&branch=main&label=test
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/test.yml
.. |publish| image:: https://img.shields.io/github/actions/workflow/status/lucasimi/tda-mapper-python/publish.yml?logo=github&logoColor=silver&label=publish
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/publish.yml
.. |docs| image:: https://img.shields.io/readthedocs/tda-mapper/main?logo=readthedocs&logoColor=silver
   :target: https://tda-mapper.readthedocs.io/en/main/
.. |codecov| image:: https://img.shields.io/codecov/c/github/lucasimi/tda-mapper-python?logo=codecov&logoColor=silver
   :target: https://codecov.io/github/lucasimi/tda-mapper-python
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281/zenodo.10642381-blue?logo=doi&logoColor=silver
   :target: https://doi.org/10.5281/zenodo.10642381
.. |Streamlit App| image:: https://img.shields.io/badge/Streamlit-App-blue?logo=streamlit&logoColor=silver
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
Mapper algorithm for Topological Data Analysis (TDA).
It enables fast computation of Mapper graphs by using *vp-trees* to optimize
the construction of open covers, improving both performance and scalability.

For further details, please refer to the
`preprint <https://doi.org/10.5281/zenodo.10659651>`__,
and 
`online documentation <https://tda-mapper.readthedocs.io/en/main/>`__.

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


Citations
---------

If you use **tda-mapper** in your work, please consider citing both the 
`library <https://doi.org/10.5281/zenodo.10642381>`__,
archived in a permanent Zenodo record, and the 
`preprint <https://doi.org/10.5281/zenodo.10659651>`__,
which provides a broader methodological overview.
We recommend citing the specific version of the library used in your research, as well as the latest version of the preprint.

- **tda-mapper**: For example to cite version 0.8.0 you can use:

  .. code:: bibtex

      @software{simi_2024_14194667,
         author       = {Simi, Luca},
         title        = {tda-mapper},
         month        = nov,
         year         = 2024,
         publisher    = {Zenodo},
         version      = {v0.8.0},
         doi          = {10.5281/zenodo.14194667},
         url          = {https://doi.org/10.5281/zenodo.14194667}
      }

- **Methodology**: For the preprint, you can use:

  .. code:: bibtex

      @misc{simi_2024_11187959,
         author       = {Simi, Luca},
         title        = {{A Scalable Approach for Mapper via Vantage Point Trees}},
         month        = may,
         year         = 2024,
         publisher    = {Zenodo},
         doi          = {10.5281/zenodo.11187959},
         url          = {https://doi.org/10.5281/zenodo.11187959}
      }


.. toctree::
   :caption: User's Guide
   :maxdepth: 1

   quickstart
   examples 


.. toctree::
   :caption: API Reference
   :maxdepth: 1

   apiref
