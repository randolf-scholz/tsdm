r"""Provides utility functions.

tsdm.util
=========

.. data:: ACTIVATIONS

    A :class:`dict` containing string names as keys and corresponding :mod:`torch` functions.

"""

from .converters import (
    make_dense_triplets,
    make_masked_format,
    make_sparse_triplets,
    time2float,
    time2int,
)
from .regularity_tests import (
    float_gcd,
    is_quasiregular,
    is_regular,
    regularity_coefficient,
    time_gcd,
)
from .util import (
    ACTIVATIONS,
    deep_dict_update,
    deep_kval_update,
    relative_error,
    scaled_norm,
    timefun,
)

__all__ = [
    "ACTIVATIONS",
    "deep_dict_update",
    "deep_kval_update",
    "relative_error",
    "scaled_norm",
    "timefun",
    "make_dense_triplets",
    "make_sparse_triplets",
    "make_masked_format",
    "time2int",
    "time2float",
    "float_gcd",
    "time_gcd",
    "regularity_coefficient",
    "is_regular",
    "is_quasiregular",
]
