r"""Provides utility functions."""
import logging
from typing import Final

from tsdm.util import (
    dataloaders,
    decorators,
    dtypes,
    regularity_tests,
    samplers,
    system,
)
from tsdm.util._norms import grad_norm, multi_norm, relative_error, scaled_norm
from tsdm.util._utilfuncs import ACTIVATIONS, deep_dict_update, deep_kval_update, now

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "ACTIVATIONS",
    "deep_dict_update",
    "deep_kval_update",
    "relative_error",
    "scaled_norm",
    "grad_norm",
    "multi_norm",
    "now",
] + [
    "dataloaders",
    "decorators",
    "dtypes",
    "regularity_tests",
    "samplers",
    "system",
]
