r"""Provides utility functions."""
import logging
from typing import Final

from tsdm.util import dataloaders, decorators, dtypes, regularity_tests, system
from tsdm.util._norms import relative_error, scaled_norm
from tsdm.util._utilfuncs import ACTIVATIONS, deep_dict_update, deep_kval_update

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "ACTIVATIONS",
    "deep_dict_update",
    "deep_kval_update",
    "relative_error",
    "scaled_norm",
] + [
    "dataloaders",
    "decorators",
    "dtypes",
    "regularity_tests",
    "system",
]
