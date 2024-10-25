"""
This module provides common functionalities for internal use.
"""
import warnings


def _deprecated(deprecated, substitute):
    msg = f'{deprecated} is deprecated and will be removed in a future version. Use {substitute} instead.'
    warnings.warn(
        msg,
        DeprecationWarning,
        stacklevel=2
    )
