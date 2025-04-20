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

   import matplotlib.pyplot as plt
   from sklearn.datasets import make_circles

   import numpy as np
   from sklearn.decomposition import PCA
   from sklearn.cluster import DBSCAN

   from tdamapper.learn import MapperAlgorithm
   from tdamapper.cover import CubicalCover
   from tdamapper.plot import MapperPlot

   # Generate toy dataset
   X, labels = make_circles(n_samples=5000, noise=0.05, factor=0.3, random_state=42)
   plt.figure(figsize=(5, 5))
   plt.scatter(X[:,0], X[:,1], c=labels, s=0.25, cmap="jet")
   plt.axis("off")
   plt.show()

   # Apply PCA as lens
   y = PCA(2, random_state=42).fit_transform(X)

   # Mapper pipeline
   cover = CubicalCover(n_intervals=10, overlap_frac=0.3)
   clust = DBSCAN()
   graph = MapperAlgorithm(cover, clust).fit_transform(X, y)

   # Visualize the Mapper graph
   fig = MapperPlot(graph, dim=2, seed=42, iterations=60).plot_plotly(colors=labels)
   fig.show(config={"scrollZoom": True})

+----------------------------------------+-----------------------------+
| Original Dataset                       | Mapper Graph                |
+========================================+=============================+
| |Original Dataset|                     | |Mapper Graph|              |
+----------------------------------------+-----------------------------+

More examples can be found in the
`documentation <https://tda-mapper.readthedocs.io/en/main/>`__.

Interactive App
---------------

Use our Streamlit app to visualize and explore your data without writing code.
You can run a live demo directly on
`Streamlit Cloud <https://tda-mapper-app.streamlit.app/>`__,
or locally on your machine using the following:

.. code:: bash

   pip install -r app/requirements.txt
   streamlit run app/streamlit_app.py


.. |Original Dataset| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_dataset_v2.png
.. |Mapper Graph| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/circles_mean_v2.png
