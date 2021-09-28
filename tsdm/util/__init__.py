r"""Provides utility functions.

TODO: Module description
"""

from __future__ import annotations

__all__ = [
    # Constants
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "relative_error",
    "scaled_norm",
    "grad_norm",
    "multi_norm",
    "now",
    # Sub-Modules
    "activations",
    "dataloaders",
    "decorators",
    "dtypes",
    "regularity_tests",
    "samplers",
    "system",
]

import logging

from tsdm.util import (
    activations,
    dataloaders,
    decorators,
    dtypes,
    regularity_tests,
    samplers,
    system,
)
from tsdm.util._norms import grad_norm, multi_norm, relative_error, scaled_norm
from tsdm.util._utilfuncs import deep_dict_update, deep_kval_update, now

LOGGER = logging.getLogger(__name__)
