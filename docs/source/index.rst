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
.. |Step 1| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_1.png
.. |Step 2| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_2.png
.. |Step 3| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_3.png
.. |Step 4| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_4.png

.. figure::
   https://github.com/lucasimi/tda-mapper-python/raw/main/docs/source/logos/tda-mapper-logo-horizontal.png
   :alt: Logo

|PyPI version| |downloads| |codecov| |test| |publish| |docs|  |DOI|

|Source Code|

tda-mapper
==========

**tda-mapper** is a Python library built around the Mapper algorithm, a core
technique in Topological Data Analysis (TDA) for extracting topological 
structure from complex data. Designed for computational efficiency and
scalability, it leverages optimized spatial search methods to support 
high-dimensional datasets. The library is well-suited for integration into
machine learning pipelines, unsupervised learning tasks, and exploratory data
analysis.

Further details in the
`documentation <https://tda-mapper.readthedocs.io/en/main/>`__
and in the
`paper <https://openreview.net/pdf?id=lTX4bYREAZ>`__.

Core features
-------------

- **Efficient construction**
    
   Leverages optimized spatial search techniques and parallelization to 
   accelerate the construction of Mapper graphs, supporting the analysis of
   high-dimensional datasets.

- **Scikit-learn integration**

   Provides custom estimators that are fully compatible with scikit-learn's
   API, enabling seamless integration into scikit-learn pipelines for tasks
   such as dimensionality reduction, clustering, and feature extraction.

- **Flexible visualization**

   Multiple visualization backends supported (Plotly, Matplotlib, PyVis) for
   generating high-quality Mapper graph representations with adjustable 
   layouts and styling.

- **Interactive app**

   Provides an interactive web-based interface for dynamic exploration of
   Mapper graph structures, offering real-time adjustments to parameters and
   visualizations.


Background
----------

The Mapper algorithm extracts topological features from complex datasets,
representing them as graphs that highlight clusters, transitions, and key
structural patterns. These insights reveal hidden data relationships and are
applicable across diverse fields, including social sciences, biology, and
machine learning. For an in-depth overview of Mapper, including its
mathematical foundations and practical applications, read 
`the original paper <https://research.math.osu.edu/tgda/mapperPBG.pdf>`__.

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
which provides a broader methodological overview. We recommend citing the
specific version of the library used in your research, along with the paper.

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
