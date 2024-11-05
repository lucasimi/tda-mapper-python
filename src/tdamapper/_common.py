"""
This module provides common functionalities for internal use.
"""
import warnings


def warn_deprecated(deprecated, substitute):
    msg = f'{deprecated} is deprecated and will be removed in a future version. Use {substitute} instead.'
    warnings.warn(
        msg,
        DeprecationWarning,
        stacklevel=2
    )


class ParamsMixin:

    def _is_param_internal(self, k):
        return k.startswith('_') or k.endswith('_')

    def get_params(self, deep=True):
        """
        Get all public parameters of the object as a dictionary.

        :param deep: A flag for returning also nested parameters.
        :type deep: bool, optional.
        """
        params = self.__dict__.items()
        return {k: v for k, v in params if not self._is_param_internal(k)}

    def set_params(self, **params):
        """
        Set public parameters. Only updates attributes that already exist.
        """
        for k, v in params.items():
            if hasattr(self, k) and not self._is_param_internal(k):
                setattr(self, k, v)
        return self


def clone(estimator):
    """
    Clone an estimator with the same parameters.

    :param estimator: An estimator to be cloned.
    :type estimator: A scikit-learn compatible estimator
    :return: A new estimator with the same parameters.
    :rtype: A scikit-learn compatible estimator
    """

    params = estimator.get_params(deep=True)
    return type(estimator)(**params)
