"""
tsdm.util
=========

Provides utility functions
"""

from .util import ACTIVATIONS, \
    deep_dict_update, deep_kval_update, \
    relative_error, scaled_norm, \
    timefun

__all__ = ['ACTIVATIONS',
           'deep_dict_update', 'deep_kval_update',
           'relative_error', 'scaled_norm',
           'timefun']
