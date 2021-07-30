r"""Provides utility functions.

tsdm.util
=========

.. data:: ACTIVATIONS

    A :class:`dict` containing string names as keys and corresponding :mod:`torch` functions.

"""

from tsdm.util import converters, dataloaders
from tsdm.util._decorators import timefun
from tsdm.util._regularity_tests import (
    float_gcd,
    is_quasiregular,
    is_regular,
    regularity_coefficient,
    time_gcd,
)
from tsdm.util.util import (
    ACTIVATIONS,
    deep_dict_update,
    deep_kval_update,
    OPTIMIZERS,
    relative_error,
    scaled_norm,
)

__all__ = [
    "ACTIVATIONS",
    "OPTIMIZERS",
    "deep_dict_update",
    "deep_kval_update",
    "relative_error",
    "scaled_norm",
    "timefun",
    "float_gcd",
    "time_gcd",
    "regularity_coefficient",
    "is_regular",
    "is_quasiregular",
] + [
    "converters",
    "dataloaders",
]
