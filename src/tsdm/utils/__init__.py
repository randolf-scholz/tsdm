r"""Provides utility functions."""

__all__ = [
    # Sub-Packages
    "data",
    "decorators",
    # Sub-Modules
    "remote",
    "system",
    # Constants
    # Classes
    "PatchedABCMeta",
    "LazyDict",
    "LazyFunction",
    # decorators
    "abstractattribute",
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "get_function_args",
    "get_mandatory_argcount",
    "initialize_from_config",
    "is_dunder",
    "is_keyword_only",
    "is_mandatory",
    "is_partition",
    "is_positional",
    "is_positional_only",
    "is_zipfile",
    "now",
    "pairwise_disjoint",
    "paths_exists",
    "prepend_path",
    "repackage_zip",
    "round_relative",
    "unflatten_dict",
]

from tsdm.utils import data, decorators, remote, system
from tsdm.utils._subclassing import PatchedABCMeta, abstractattribute
from tsdm.utils._utils import (
    deep_dict_update,
    deep_kval_update,
    flatten_dict,
    flatten_nested,
    get_function_args,
    get_mandatory_argcount,
    initialize_from_config,
    is_dunder,
    is_keyword_only,
    is_mandatory,
    is_partition,
    is_positional,
    is_positional_only,
    is_zipfile,
    now,
    pairwise_disjoint,
    paths_exists,
    prepend_path,
    repackage_zip,
    round_relative,
    unflatten_dict,
)
from tsdm.utils.lazydict import LazyDict, LazyFunction
