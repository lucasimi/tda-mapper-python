import tdamapper.utils.cython.metrics as cython_metrics


def get_metric(metric, **kwargs):
    return cython_metrics.get_metric(metric, **kwargs)
