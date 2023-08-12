from mapper.core import build_labels
from mapper.cover import TrivialCover


class TrivialClustering:

    def TrivialClustering(self):
        pass

    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **parameters):
        return self

    def _set_n_features_in_(self, X):
        self.n_features_in_ = len(X[0])

    def _check_input(self, X, y):
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
        return X, y

    def fit(self, X, y=None):
        X, y = self._check_input(X, y)
        self.labels_ = [0 for _ in X]
        self._set_n_features_in_(X)
        return self


class CoverClustering:

    def __init__(self, cover=None):
        self.cover = cover

    def _check_params(self):
        if not self.cover:
            cover = TrivialCover()
        else:
            cover = self.cover
        return cover

    def get_params(self, deep=True):
        parameters = {}
        parameters['cover'] = self.cover
        if deep:
            if self.cover:
                for k, v in self.cover.get_params().items():
                    parameters[f'cover__{k}'] = v
        return parameters
    
    def set_params(self, **parameters):
        for k, v in parameters.items():
            setattr(self, k, v)
        return self

    def _set_n_features_in_(self, X):
        self.n_features_in_ = len(X[0])

    def _check_input(self, X, y):
        if y is None:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)
        return X, y

    def fit(self, X, y=None):
        X, y = self._check_input(X, y)
        if self.cover:
            cover = self.cover
        else:
            cover = TrivialCover()
        multilabels = build_labels(X, X, cover, TrivialClustering())
        label_values = set()
        for labels in multilabels:
            label_values.update(labels)
        uf = UnionFind(label_values)
        self.labels_ = []
        for labels in multilabels:
            if len(labels) > 1:
                for first, second in zip(labels, labels[1:]):
                    root = uf.union(first, second)
            else:
                root = uf.find(labels[0])
            self.labels_.append(root)
        self._set_n_features_in_(X)
        return self
