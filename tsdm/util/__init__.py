"""
tsdm.util
=========

Provides utility functions

.. data:: ACTIVATIONS

    A :class:`dict` containing string names as keys and corresponding :mod:`torch` functions.

"""

from .util import ACTIVATIONS, \
    deep_dict_update, deep_kval_update, \
    relative_error, scaled_norm, \
    timefun

from .converters import make_dense_triplets, make_sparse_triplets, make_masked_format

__all__ = ['ACTIVATIONS',
           'deep_dict_update', 'deep_kval_update',
           'relative_error', 'scaled_norm',
           'timefun',
           'make_dense_triplets', 'make_sparse_triplets', 'make_masked_format']
