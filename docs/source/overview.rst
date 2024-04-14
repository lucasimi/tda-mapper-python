Overview
========

**tda-mapper** is a simple and efficient implementation of Mapper algorithm for
Topological Data Analysis.

-  **Installation**: ``pip install tda-mapper``

-  **Documentation**: https://tda-mapper.readthedocs.io/en/main/

-  **Demo App**: https://tda-mapper-app.streamlit.app/

The Mapper algorithm is a well-known technique in the field of
topological data analysis that allows data to be represented as a graph.
Mapper is used in various fields such as machine learning, data mining,
and social sciences, due to its ability to preserve topological features
of the underlying space, providing a visual representation that
facilitates exploration and interpretation. For an in-depth coverage of
Mapper you can read 
`the original paper <https://research.math.osu.edu/tgda/mapperPBG.pdf>`__.

This library contains an implementation of Mapper, where the construction 
of open covers is based on *vp-trees* for improved performance and scalability.
The details about this methodology are contained in
`our preprint <https://doi.org/10.5281/zenodo.10659652>`__.

+-----------------+-----------------+-----------------+-----------------+
| Step 1          | Step 2          | Step 3          | Step 4          |
+=================+=================+=================+=================+
| |Step 1|        | |Step 2|        | |Step 3|        | |Step 4|        |
+-----------------+-----------------+-----------------+-----------------+

.. |Step 1| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_1.png
.. |Step 2| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_2.png
.. |Step 3| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_3.png
.. |Step 4| image:: https://github.com/lucasimi/tda-mapper-python/raw/main/resources/mapper_4.png

