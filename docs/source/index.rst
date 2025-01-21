.. |Source Code| image:: https://img.shields.io/badge/lucasimi-tda--mapper--python-blue?logo=github&logoColor=silver
   :target: https://github.com/lucasimi/tda-mapper-python
.. |PyPI version| image:: https://img.shields.io/pypi/v/tda-mapper?logo=python&logoColor=silver
   :target: https://pypi.python.org/pypi/tda-mapper
.. |downloads| image:: https://img.shields.io/pypi/dm/tda-mapper?logo=python&logoColor=silver
   :target: https://pypi.python.org/pypi/tda-mapper
.. |test| image:: https://img.shields.io/github/actions/workflow/status/lucasimi/tda-mapper-python/test-unit.yml?logo=github&logoColor=silver&branch=main&label=test
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/test-unit.yml
.. |publish| image:: https://img.shields.io/github/actions/workflow/status/lucasimi/tda-mapper-python/publish-pypi.yml?logo=github&logoColor=silver&label=publish
   :target: https://github.com/lucasimi/tda-mapper-python/actions/workflows/publish-pypi.yml
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

|Source Code| |PyPI version| |downloads| |test| |publish| |docs| |codecov| |DOI|
|Streamlit App|

tda-mapper
==========

**tda-mapper** is a Python library based on the Mapper algorithm, a key tool in
Topological Data Analysis (TDA). Designed for efficient computations and backed
by advanced spatial search techniques, it scales seamlessly to high dimensional
data, making it suitable for applications in machine learning, data mining, and
exploratory data analysis.

Further details in the
`documentation <https://tda-mapper.readthedocs.io/en/main/>`__
and in the
`paper <https://openreview.net/pdf?id=lTX4bYREAZ>`__.

Main features
-------------

- **Fast Mapper graph construction**: Accelerates computations with efficient
  spatial search, enabling analysis of large, high-dimensional datasets.

- **Scikit-learn compatibility**: Easily integrate Mapper as a part of your
  machine learning workflows.

- **Flexible visualization options**: Visualize Mapper graphs with multiple
  supported backends, tailored to your needs.

- **Interactive exploration**: Explore data interactively through a
  user-friendly app.

Background
----------

The Mapper algorithm transforms complex datasets into graph representations
that highlight clusters, transitions, and topological features. These insights
reveal hidden patterns in data, applicable across fields like social sciences,
biology, and machine learning. For an in-depth coverage of Mapper, including
its mathematical foundations and applications, read the 
`original paper <https://research.math.osu.edu/tgda/mapperPBG.pdf>`__.

+-----------------+-----------------+-----------------+-----------------+
| Step 1          | Step 2          | Step 3          | Step 4          |
+=================+=================+=================+=================+
| |Step 1|        | |Step 2|        | |Step 3|        | |Step 4|        |
+-----------------+-----------------+-----------------+-----------------+
| Choose lens     | Cover image     | Run clustering  | Build graph     |
+-----------------+-----------------+-----------------+-----------------+

Citations
---------

If you use **tda-mapper** in your work, please consider citing both the 
`library <https://doi.org/10.5281/zenodo.10642381>`__,
archived in a permanent Zenodo record, and the 
`paper <https://openreview.net/pdf?id=lTX4bYREAZ>`__,
which provides a broader methodological overview.
We recommend citing the specific version of the library used in your research, as well as the paper.

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

- **Methodology**: For the paper, you can use:

  .. code:: bibtex

      @article{simi2025a,
         title    = {A Scalable Approach for Mapper via Efficient Spatial Search},
         author   = {Luca Simi},
         journal  = {Transactions on Machine Learning Research},
         issn     = {2835-8856},
         year     = {2025},
         url      = {https://openreview.net/forum?id=lTX4bYREAZ},
         note     = {}
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
