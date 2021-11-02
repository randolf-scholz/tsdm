r"""Provides utility functions.

TODO: Module description
"""

from __future__ import annotations

__all__ = [
    # Sub-Modules
    "activations",
    "dataloaders",
    "decorators",
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
    "grad_norm",
    "initialize_from",
    "multi_norm",
    "now",
    "relative_error",
    "relsize_skewpart",
    "relsize_symmpart",
    "scaled_norm",
    "skew_symmetric",
    "symmetric",
]

import logging

from tsdm.util import (
    activations,
    dataloaders,
    decorators,
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
    relsize_skewpart,
    relsize_symmpart,
    skew_symmetric,
    symmetric,
)

__logger__ = logging.getLogger(__name__)
