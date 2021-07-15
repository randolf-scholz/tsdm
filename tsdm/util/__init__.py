r"""Provides utility functions.

tsdm.util
=========

.. data:: ACTIVATIONS

    A :class:`dict` containing string names as keys and corresponding :mod:`torch` functions.

"""

import tsdm.util.dataloaders as dataloaders
import tsdm.util.converters as converters

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
    "dataloaders",
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
]
