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

    def get_params(self, deep=True):
        """
        Get all public parameters of the object as a dictionary.

        :param deep: A flag for returning also nested parameters.
        :type deep: bool, optional.
        """
        params = self.__dict__.items()
        return {k: v for k, v in params if not k.startswith('_')}

    def set_params(self, **params):
        """
        Set public parameters. Only updates attributes that already exist.
        """
        for k, v in params.items():
            if hasattr(self, k) and not k.startswith('_'):
                setattr(self, k, v)
        return self
