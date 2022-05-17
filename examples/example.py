import numpy as np
from sklearn.cluster import DBSCAN

from mapper.cover import SearchCover
from mapper.search import BallSearch
from mapper.pipeline import MapperPipeline

import matplotlib.pyplot as plt


mp = MapperPipeline(
    cover_algo=SearchCover(
        search_algo=BallSearch(1.5), 
        metric=lambda x, y: np.linalg.norm(x - y), 
        lens=lambda x: x),
    clustering_algo=DBSCAN(eps=1.5, min_samples=2)
)

data = [np.random.rand(10) for _ in range(100)]
g = mp.fit(data)

g.colorize(data, np.nanmean)
fig1 = g.plot(512, 512, frontend='plotly')
fig1.show()

fig2 = g.plot(512, 512, frontend='matplotlib')
plt.show()