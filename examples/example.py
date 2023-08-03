from sklearn.datasets import load_iris

from mapper.core import *
from mapper.cover import *
from mapper.clustering import *
from mapper.plot import *

iris_data = load_iris()
iris_data
X = iris_data.data[:, :]
y = iris_data.target

cover_algo = BallCover(radius=1.0, metric=lambda x, y: np.linalg.norm(x - y))
mapper_algo = MapperAlgorithm(cover=cover_algo, clustering=TrivialClustering())
mapper_graph = mapper_algo.build_graph(X)
mapper_plot = MapperPlot(X, mapper_graph)

fig1 = mapper_plot.with_colors().plot('plotly', 512, 512, 'mean lens')
fig1.show()

fig2 = mapper_plot.with_colors(colors=list(y)).plot('plotly', 512, 512, 'mean lens')
fig2.show()
