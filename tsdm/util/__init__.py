r"""Provides utility functions.

tsdm.util
=========

.. data:: ACTIVATIONS

    A :class:`dict` containing string names as keys and corresponding :mod:`torch` functions.

"""

from tsdm.util import converters
from tsdm.util import dataloaders
from tsdm.util._regularity_tests import (
    float_gcd,
    is_quasiregular,
    is_regular,
    regularity_coefficient,
    time_gcd,
)
from tsdm.util._util import (
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
    "float_gcd",
    "time_gcd",
    "regularity_coefficient",
    "is_regular",
    "is_quasiregular",
] + [
    "converters",
    "dataloaders",
]
