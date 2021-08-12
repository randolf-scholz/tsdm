r"""Provides utility functions."""
import logging
from typing import Final

from tsdm.util import dataloaders, decorators, regularity_tests
from tsdm.util._utilfuncs import (
    ACTIVATIONS,
    deep_dict_update,
    deep_kval_update,
    relative_error,
    scaled_norm,
    TimeDeltaLike,
    TimeStampLike,
)

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "ACTIVATIONS",
    "deep_dict_update",
    "deep_kval_update",
    "relative_error",
    "scaled_norm",
    "TimeStampLike",
    "TimeDeltaLike",
] + [
    "dataloaders",
    "decorators",
    "regularity_tests",
]
