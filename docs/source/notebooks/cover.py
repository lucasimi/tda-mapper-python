# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %% [markdown]

# # Example 3: Exploring Cover Algorithms

# In this notebook, we make a comparison among all the cover algorithms offered by this library
# with the goal of offering some guidance on how to choose the best cover algorithm for your
# specific dataset and analysis goals. Each algorithm captures different aspects of the data, and
# the choice of cover can significantly influence the resulting Mapper graph. It is important to
# experiment with different cover algorithms and parameters to find the best fit for your specific
# dataset and analysis goals. The choice of cover algorithm can reveal different patterns and
# structures in the data, and understanding these differences can help you gain deeper insights
# into the underlying data distribution and relationships. It's important to remind that the cover
# algorithm is applied to the lens data. So whenever we use the word "space" in this notebook, we
# are referring to the lens data.

# We will use the **Digits dataset** as a case study, applying different cover algorithms to see
# how they affect the Mapper graph structure. The goal is to understand how different covers can
# reveal various aspects of the data and how they can be used to highlight different features in
# the Mapper analysis. In the following examples we will skip the clustering step, as we are
# interested in the cover algorithms only.

# %%
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from tdamapper.learn import MapperAlgorithm
from tdamapper.plot import MapperPlot

X, labels = load_digits(return_X_y=True)
y = PCA(2, random_state=42).fit_transform(X)


def mode(arr):
    values, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    mode_values = values[counts == max_count]
    return np.nanmean(mode_values)


# %% [markdown]

# ## 1. CubicalCover

# The `CubicalCover` covers the space into a grid-like structure, where each cell has a fixed size
# and overlaps with adjacent cells.

# ### Parameters

# The `n_intervals` parameter controls the number of intervals in each dimension, and
# `overlap_frac` controls the overlap between adjacent intervals. You can adjust these parameters
# to see how they affect the Mapper graph. A larger number of intervals and a smaller overlap
# fraction will create a finer cover, potentially revealing more detail in the data, but also
# possibly introducing noise. Conversely, a smaller number of intervals with a larger overlap
# fraction will create a coarser cover, which may smooth out some of the finer details but can also
# help to reduce noise and highlight broader patterns. The choice of these parameters can
# significantly influence the structure of the Mapper graph, so it's important to experiment with
# different values to find the best fit for your data.

# Additionally, the `CubicalCover` has an `algorithm` parameter that allows you to choose between
# different algorithms for creating the cover. The default algorithm is `proximity`, which creates
# a grid that is enough to cover the dataset. However, you can also choose `standard`, which
# creates a grid which contains all the cells that cover the dataset. The `proximity` algorithm is
# the default because it is more efficient and scales well in high dimensions, while the`standard`
# algorithm is more straightforward and it's consistent with the original Mapper algorithm described
# in the original paper. The choice of algorithm can affect the structure of the Mapper graph,
# especially in high-dimensional spaces, where the `proximity` algorithm can help to reduce the
# computational complexity and improve the performance of the Mapper analysis, producing more
# compact graphs with lower noise. The `standard` algorithm, on the other hand, can produce more
# detailed graphs, as it captures all the cells that cover the dataset, but it may also introduce
# more noise and complexity, especially in high-dimensional spaces where the number of cells can
# grow exponentially with the number of dimensions, making it more usable in low-dimensional
# spaces.

# ### Advantages
# One advantage of using `CubicalCover` is that it is the most widely used cover algorithm in
# Mapper analysis, and it is often the default choice in many Mapper implementations. It is
# computationally efficient, as it does not require calculating distances between all pairs of
# points, and it can work well in high-dimensional spaces. It's also the default choice in many
# research papers and applications, making it a familiar and well-understood method for many
# researchers and practitioners.

# ### Disadvantages
# One disadvantage of using `CubicalCover` is that it can be sensitive to the choice of parameters,
# particularly the number of intervals and the overlap fraction. If these parameters are not
# chosen carefully, the resulting Mapper graph may not accurately reflect the underlying data
# distribution. For example, if the number of intervals is too small or the overlap fraction is
# too large, the cover may merge distinct clusters or fail to capture important local structures in
# the data. This can lead to a loss of information and make it difficult to interpret the
# resulting Mapper graph. Therefore, it is important to carefully choose the parameters and
# consider the specific characteristics of the dataset when using `CubicalCover`.


# %%
from tdamapper.cover import CubicalCover

mapper = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.25,
    ),
    verbose=False,
)

graph = mapper.fit_transform(X, y)

plot = MapperPlot(graph, dim=3, iterations=400, seed=42)

fig = plot.plot_plotly(
    colors=labels,
    cmap=["jet", "viridis", "cividis"],
    agg=mode,
    node_size=[0.25 * x for x in range(9)],
    title="mode of digits",
)

fig.show(config={"scrollZoom": True}, renderer="notebook_connected")

# %%
from tdamapper.cover import CubicalCover

mapper = MapperAlgorithm(
    cover=CubicalCover(
        n_intervals=10,
        overlap_frac=0.25,
        algorithm="standard",
    ),
    verbose=False,
)

graph = mapper.fit_transform(X, y)

plot = MapperPlot(graph, dim=3, iterations=400, seed=42)

fig = plot.plot_plotly(
    colors=labels,
    cmap=["jet", "viridis", "cividis"],
    agg=mode,
    node_size=[0.25 * x for x in range(9)],
    title="mode of digits",
)

fig.show(config={"scrollZoom": True}, renderer="notebook_connected")

# %% [markdown]

# ## 2. BallCover

# The `BallCover` algorithm creates a cover based on balls of a specified radius around points.

# ### Parameters
# The key parameters in the `BallCover` is the `radius`, which determines the size of the balls
# used to cover the space. A larger radius will create a coarser cover, potentially merging nearby
# points into the same ball, while a smaller radius will create a finer cover, allowing for more
# detailed local structures to be captured. The choice of radius can significantly affect the
# resulting Mapper graph, as it determines how points are grouped together and how the overall
# structure of the data is represented. Another important parameter is the `metric`, which defines
# the distance function used to measure the distance between points. The default metric is
# Euclidean distance, but you can also use other metrics such as cosine distance or Manhattan
# distance, depending on the characteristics of your dataset and the specific analysis goals. The
# choice of metric can also influence the structure of the Mapper graph, as different metrics may
# capture different aspects of the data distribution and relationships between points.

# ### Advantages
# One advantage of using `BallCover` is that it's computationally efficient, as it does not
# require calculating distances between all pairs of points, being based on an efficient indexing
# of the space. Moreover, it can work on any metric space, as it only requires a distance function
# to define the balls. This makes it a versatile choice for many different types of datasets.

# ### Disadvantages
# One disadvantage of using `BallCover` is that it can be sensitive to the density of points in the
# dataset. In regions with high point density, the balls may overlap significantly, leading to a
# more complex graph structure. In contrast, in regions with low point density, the balls may not
# overlap much, resulting in isolated nodes or small clusters. Using the same radius for the entire
# dataset may not capture the local structure effectively and this can lead to a loss of
# information and make it difficult to interpret the resulting Mapper graph. Therefore, it is
# important to carefully choose the radius and consider the density of points in the dataset when
# using `BallCover`. A good choice of radius can help to balance the trade-off between capturing
# local structures and avoiding noise in the Mapper graph. In practice, it may be beneficial to
# experiment with different radius values and analyze the resulting Mapper graphs to find the
# optimal radius for a given dataset. Chosing a good radius can be tricky especially in
# high-dimensional spaces, where the distance between points can become less meaningful due to the
# curse of dimensionality. In such cases, it may be beneficial to use a cover that adapts to the
# local density of points, such as the `KNNCover`, which uses k-nearest neighbors to define the
# cover.

# %%
from tdamapper.cover import BallCover

mapper = MapperAlgorithm(
    cover=BallCover(radius=5.0),
    verbose=False,
)

graph = mapper.fit_transform(X, y)

plot = MapperPlot(graph, dim=3, iterations=400, seed=42)

fig = plot.plot_plotly(
    colors=labels,
    cmap=["jet", "viridis", "cividis"],
    agg=mode,
    node_size=[0.25 * x for x in range(9)],
    title="mode of digits",
)

fig.show(config={"scrollZoom": True}, renderer="notebook_connected")

# %% [markdown]

# ## 3. KNNCover

# The `KNNCover` algorithm uses k-nearest neighbors to define the cover. The cover is created by
# choosing a set of points in the dataset and then connecting each point to its k-nearest
# neighbors. For this reason, each set in the cover has cardinality equal to the number of
# neighbors specified.

# ### Parameters
# The key parameter in the `KNNCover` is the `neighbors`, which determines how many nearest
# neighbors to consider when creating the cover. A larger number of neighbors will create a more
# connected cover, potentially capturing more global structure, while a smaller number of neighbors
# will create a more localized cover, focusing on the immediate neighborhood of each point. The
# choice of the number of neighbors can significantly affect the resulting Mapper graph, as it
# determines how points are grouped together and how the overall structure of the data is
# represented.

# ### Advantages
# One advantage of using `KNNCover` is that it can adapt to the local density of points, allowing
# for a more nuanced representation of the data. In regions with high point density, the cover will
# create more sets of points, while in regions with low point density, the cover will create fewer
# sets. This can help to reveal local structures and patterns that may not be visible with other
# cover methods. Similarly to `BallCover`, it is also computationally efficient, as it does not
# require calculating distances between all pairs of points, being based on an efficient indexing
# of the space. Moreover, it can work on any metric space, as it only requires a distance function
# to define the nearest neighbors. This makes it a versatile choice for many different types of
# datasets.

# ### Disadvantages
# One possible disadvantage of using `KNNCover` is that chosing the number of neighbors can
# significantly affect the resulting Mapper graph. If the number of neighbors is too small, the
# cover may not capture the global structure of the data, leading to a fragmented graph with many
# isolated nodes or small clusters. On the other hand, if the number of neighbors is too large, the
# cover may merge distinct clusters or fail to capture important local structures in the data.
# Additionally, small isolated clusters (smaller than the number of neighbors) may not be captured
# effectively. If a cluster is smaller than the number of neighbors specified, it may be merged with
# other larger clusters, leading to a loss of information about the smaller cluster.

# %%
from tdamapper.cover import KNNCover

mapper = MapperAlgorithm(
    cover=KNNCover(neighbors=15),
    verbose=False,
)

graph = mapper.fit_transform(X, y)

plot = MapperPlot(graph, dim=3, iterations=400, seed=42)

fig = plot.plot_plotly(
    colors=labels,
    cmap=["jet", "viridis", "cividis"],
    agg=mode,
    node_size=[0.125 * x for x in range(9)],
    title="mode of digits",
)

fig.show(config={"scrollZoom": True}, renderer="notebook_connected")

# %% [markdown]

# ## Conclusions

# In this notebook, we explored three different cover algorithms: `CubicalCover`, `BallCover`, and
# `KNNCover`. Each algorithm has its own strengths and weaknesses, and the choice of cover can
# significantly influence the resulting Mapper graph. Here is a summary of the key differences
# between the three cover algorithms:

# +------------------------+------------------+-------------------------------------+-------------------------------------+
# | Cover Algorithm        | Parameters       | Advantages                          | Disadvantages                       |
# +========================+==================+=====================================+=====================================+
# | CubicalCover           | - `n_intervals`  | - Widely used and well-supported    | - Sensitive to parameters           |
# |                        | - `overlap_frac` | - Easy to interpret                 | - Only supports Euclidean spaces    |
# |                        | - `algorithm`    |                                     |                                     |
# +------------------------+------------------+-------------------------------------+-------------------------------------+
# | BallCover              | - `radius`       | - Works with any metric space       | - Struggles with varying densities  |
# |                        | - `metric`       | - Can capture isolated clusters     | - Radius tuning can be difficult    |
# +------------------------+------------------+-------------------------------------+-------------------------------------+
# | KNNCover               | - `neighbors`    | - Works with any metric space       | - Struggles with isolated clusters  |
# |                        | - `metric`       | - Adapts to local densities         | - Risk of over-connecting nodes     |
# +------------------------+------------------+-------------------------------------+-------------------------------------+

# As a final remark, in the example dataset that we used, despite a significative difference in the
# structure of the Mapper graph, the relationship between the different parts of the data are still
# preserved. This means that even though the cover algorithms create different structures, they
# still capture the same underlying relationships between the data points. This is an important
# aspect of Mapper analysis, as it allows for flexibility in choosing the cover algorithm while
# still maintaining the integrity of the data relationships. In practice, it is often beneficial to
# try multiple cover algorithms and compare the resulting Mapper graphs to gain a comprehensive
# understanding of the data.
