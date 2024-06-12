from numpy.linalg import norm


def euclidean():
    return minkowski(2)


def chebyshev():
    def _cheb(x, y):
        max(abs(x - y))
    return _cheb


def minkowski(p=2):
    def _minkowski(x, y):
        return norm(x - y, ord=p)
    return _minkowski


def get_metric(metric, **kwargs):
    if isinstance(metric, str):
        if metric == 'euclidean':
            return euclidean()
        elif metric == 'minkowski':
            if 'p' in kwargs:
                return minkowski(p=kwargs['p'])
            else:
                return minkowski()
        elif metric == 'chebyshev':
            return chebyshev()
    elif callable(metric):
        return metric
    else:
        raise ValueError('metric must be a string or callable')
