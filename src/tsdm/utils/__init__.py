r"""Provides utility functions."""

__all__ = [
    # Sub-Modules
    "remote",
    "funcutils",
    "lazydict",
    # Constants
    # Classes
    "timer",
    # utils
    "normalize_axes",
    "deep_dict_update",
    "normalize_dimarg",
    "flatten_dict",
    "flatten_nested",
    "last",
    "timestamp",
    "timedelta",
    "nested_paths_exist",
    "repackage_zip",
    "replace",
    "transpose_list_of_dicts",
    "unflatten_dict",
]

from . import funcutils, lazydict, remote
from ._utils import (
    deep_dict_update,
    flatten_dict,
    flatten_nested,
    last,
    nested_paths_exist,
    normalize_axes,
    normalize_dimarg,
    repackage_zip,
    replace,
    timedelta,
    timestamp,
    transpose_list_of_dicts,
    unflatten_dict,
)
from .timer import timer
