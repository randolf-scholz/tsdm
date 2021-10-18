r"""Provides utility functions.

TODO: Module description
"""

from __future__ import annotations

__all__ = [
    # Sub-Modules
    "activations",
    "dataloaders",
    "decorators",
    "dtypes",
    "regularity_tests",
    "samplers",
    "system",
    "types",
    # Constants
    # Classes
    "Split",
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "multi_norm",
    "now",
    "initialize_from",
    "grad_norm",
    "relative_error",
    "scaled_norm",
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
    types,
)
from tsdm.util._norms import grad_norm, multi_norm, relative_error, scaled_norm
from tsdm.util._util import (
    Split,
    deep_dict_update,
    deep_kval_update,
    initialize_from,
    now,
)

LOGGER = logging.getLogger(__name__)
