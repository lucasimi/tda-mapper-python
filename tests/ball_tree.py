from sklearn.neighbors import BallTree


class SkBallTree:

    def __init__(
        self,
        X,
        metric="euclidean",
        leaf_capacity=1,
        leaf_radius=0.0,
        pivoting=None,
        **kwargs,
    ):
        self.__dataset = X
        self.__ball_tree = BallTree(
            X,
            leaf_size=leaf_capacity,
            metric=metric,
            **kwargs,
        )

    def ball_search(self, point, eps, inclusive=True):
        ids = self.__ball_tree.query_radius(
            [point],
            eps,
            return_distance=False,
            count_only=False,
            sort_results=False,
        )
        return [self.__dataset[i] for i in ids[0]]

    def knn_search(self, point, k):
        ids = self.__ball_tree.query(
            [point],
            k=k,
            return_distance=False,
            dualtree=False,
            breadth_first=False,
        )
        return [self.__dataset[i] for i in ids[0]]
