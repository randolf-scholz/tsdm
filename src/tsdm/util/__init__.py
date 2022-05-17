r"""Provides utility functions."""

__all__ = [
    # Sub-Modules
    "dataloaders",
    "decorators",
    "regularity_tests",
    "system",
    "types",
    "torch",
    # Constants
    # Classes
    "Split",
    "LazyDict",
    "LazyFunction",
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "grad_norm",
    "initialize_from",
    "initialize_from_config",
    "is_partition",
    "multi_norm",
    "now",
    "paths_exists",
    "prepend_path",
    "relative_error",
    "relsize_skewpart",
    "relsize_symmpart",
    "round_relative",
    "scaled_norm",
    "skewpart",
    "symmpart",
]

import logging

from tsdm.util import dataloaders, decorators, regularity_tests, system, torch, types
from tsdm.util._norms import grad_norm, multi_norm, relative_error, scaled_norm
from tsdm.util._util import (
    Split,
    deep_dict_update,
    deep_kval_update,
    flatten_dict,
    flatten_nested,
    initialize_from,
    initialize_from_config,
    is_partition,
    now,
    paths_exists,
    prepend_path,
    relsize_skewpart,
    relsize_symmpart,
    round_relative,
    skewpart,
    symmpart,
)
from tsdm.util.lazydict import LazyDict, LazyFunction

__logger__ = logging.getLogger(__name__)
