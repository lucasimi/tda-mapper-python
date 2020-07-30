import numpy as np
import statistics
from mapper import *
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

def ball_mapper_example():
    dataset, _ = make_blobs(n_samples=5000, centers=[[-1, -1], [1, -1], [1, 1]], cluster_std=0.3, random_state=0)

    m = BallMapper(distance=lambda x,y: np.linalg.norm(x - y), radius=0.4)    
    m.fit(dataset, alg=DBSCAN(eps=0.2, min_samples=20))
    
    m.plot(aggfunc=statistics.mean, colors=[0.0 for x in dataset], colormap=lambda x: 'red')
    
def mapper_example():
    # dataset is expected to be a list of np arrays
    dataset, _ = make_blobs(n_samples=5000, centers=[[-1, -1], [1, -1], [1, 1]], cluster_std=0.3, random_state=0)
    
    # Create a mapper object with specified distance function, and radius
    m = Mapper(distance=lambda x, y: abs(x - y), radius=0.4)

    # Fit a dataset with the current mapper settings and arguments: 
    # 1. lens is expected to either be a lambda or a list (one value for each point of dataset)
    # 2. alg is expected to be any object with a fit method and labels_ attribute 
    m.fit(dataset, lens=lambda x: x[0] + x[1], alg=DBSCAN(eps=0.2, min_samples=20))

    # Color the mapper graph:
    # 1. aggfunc is any aggregation function from a list of values to a single value
    # 2. colors is expected to either be a lambda or a list (one value for each point of dataset)
    # 3. colormap is a lambda from a value to an rgb color (values accepted as #rrggbb or by literal)
    m.plot(aggfunc=statistics.mean, colors=lambda x: 0.0, colormap=lambda x: 'red')
    
ball_mapper_example()
