r"""Provides utility functions."""

__all__ = [
    # Sub-Modules
    "dataloaders",
    "decorators",
    "remote",
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
    "initialize_from",
    "initialize_from_config",
    "is_partition",
    "now",
    "paths_exists",
    "prepend_path",
    "round_relative",
]

import logging

from tsdm.util import dataloaders, decorators, remote, system, torch, types
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
    round_relative,
)
from tsdm.util.lazydict import LazyDict, LazyFunction

__logger__ = logging.getLogger(__name__)
