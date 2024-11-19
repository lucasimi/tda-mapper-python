Quick Start
===========


Installation
------------

To install the latest version uploaded on PyPI

.. code:: bash

   pip install tda-mapper


Development
-----------

- To install the latest version with dev dependencies

.. code:: bash

   pip install tda-mapper[dev]

- To install from the latest commit on main branch

.. code:: bash

   pip install git+https://github.com/lucasimi/tda-mapper-python

- To install from the latest commit of develop branch

.. code:: bash

   pip install git+https://github.com/lucasimi/tda-mapper-python@develop


How To Use
----------

Here's a minimal example using the **circles dataset** from
``scikit-learn`` to demonstrate how to use **tda-mapper**:

.. code:: python

   import numpy as np

   from sklearn.datasets import make_circles
   from sklearn.decomposition import PCA
   from sklearn.cluster import DBSCAN

   from tdamapper.core import MapperAlgorithm
   from tdamapper.cover import CubicalCover
   from tdamapper.plot import MapperPlot

   # load a labelled dataset
   X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)
   y = PCA(2, random_state=42).fit_transform(X)

   cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
   clust = DBSCAN()
   graph = MapperAlgorithm(cover, clust).fit_transform(X, y)

   # color according to labels
   fig = MapperPlot(graph, dim=2, seed=42, iterations=60).plot_plotly(colors=labels)
   fig.show(config={'scrollZoom': True})

+----------------------------------------+-----------------------------+
| Original Dataset                       | Mapper Graph                |
+========================================+=============================+
| |Original Dataset|                     | |Mapper Graph|              |
+----------------------------------------+-----------------------------+

More examples can be found in the
`documentation <https://tda-mapper.readthedocs.io/en/main/>`__.

Interactive App
---------------

You can explore a live demo of **tda-mapper** directly on
`Streamlit Cloud <https://tda-mapper-app.streamlit.app/>`__,
or run it locally using the following:

.. code:: bash

   pip install -r app/requirements.txt
   streamlit run app/streamlit_app.py


.. |Original Dataset| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_dataset.png
.. |Mapper Graph| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_mean.png
