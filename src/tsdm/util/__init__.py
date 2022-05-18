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
    "reldist_skew",
    "reldist_symm",
    "round_relative",
    "scaled_norm",
    "skewpart",
    "symmpart",
    "erank",
    "col_corr",
    "row_corr",
    "mat_corr",
    "reldist_diag",
    "orthpart",
    "reldist_orth",
]

import logging

from tsdm.util import dataloaders, decorators, regularity_tests, system, torch, types
from tsdm.util._norms import grad_norm, multi_norm, relative_error, scaled_norm
from tsdm.util._util import (
    Split,
    col_corr,
    deep_dict_update,
    deep_kval_update,
    erank,
    flatten_dict,
    flatten_nested,
    initialize_from,
    initialize_from_config,
    is_partition,
    mat_corr,
    now,
    orthpart,
    paths_exists,
    prepend_path,
    reldist_diag,
    reldist_orth,
    reldist_skew,
    reldist_symm,
    round_relative,
    row_corr,
    skewpart,
    symmpart,
)
from tsdm.util.lazydict import LazyDict, LazyFunction

__logger__ = logging.getLogger(__name__)
