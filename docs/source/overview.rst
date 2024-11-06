Overview
========

**tda-mapper** is a simple and efficient Python library implementing the Mapper algorithm for Topological Data Analysis (TDA).
It enables fast computation of Mapper graphs using *vp-trees* to optimize the construction of open covers for enhanced performance and scalability.

For further details, please refer to our 
`preprint <https://doi.org/10.5281/zenodo.10659651>`__.

- **Installation**: ``pip install tda-mapper``

- **Documentation**: https://tda-mapper.readthedocs.io/en/main/

Features
--------

- **Efficient Mapper Computation**: Optimized for higher-dimensional lenses.

- **Interactive Visualizations**: Multiple plotting backends for flexibility.

- **Data Exploration App**: Interactive tool for quick, in-depth data exploration.

Demo App
--------

To assess the features of **tda-mapper** you can start from the demo app.

- **Live demo:** https://tda-mapper-app.streamlit.app/

- **Run locally:** use the following commands

  .. code:: bash

    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py

The Mapper algorithm is a well-known technique in the field of
topological data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining,
and social sciences, due to its ability to preserve topological features
of the underlying space, providing a visual representation that
facilitates exploration and interpretation. For an in-depth coverage of
Mapper you can read 
`the original paper <https://research.math.osu.edu/tgda/mapperPBG.pdf>`__.
