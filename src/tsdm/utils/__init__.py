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
    "LazyValue",
    # decorators
    "abstractattribute",
    # utils
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "initialize_from_config",
    "is_dunder",
    "is_zipfile",
    "now",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
    "paths_exists",
    "prepend_path",
    "repackage_zip",
    "round_relative",
    "unflatten_dict",
    # funcutils
    "accepts_varkwargs",
    "dataclass_args_kwargs",
    "get_function_args",
    "get_parameter_kind",
    "is_keyword_arg",
    "is_keyword_only_arg",
    "is_mandatory_arg",
    "is_positional_arg",
    "is_positional_only_arg",
    "is_variadic_arg",
    "get_mandatory_argcount",
    "get_mandatory_kwargs",
]

from tsdm.utils import data, decorators, remote, system
from tsdm.utils._subclassing import PatchedABCMeta, abstractattribute
from tsdm.utils._utils import (
    deep_dict_update,
    deep_kval_update,
    flatten_dict,
    flatten_nested,
    initialize_from_config,
    is_dunder,
    is_zipfile,
    now,
    pairwise_disjoint,
    pairwise_disjoint_masks,
    paths_exists,
    prepend_path,
    repackage_zip,
    round_relative,
    unflatten_dict,
)
from tsdm.utils.funcutils import (
    accepts_varkwargs,
    dataclass_args_kwargs,
    get_function_args,
    get_mandatory_argcount,
    get_mandatory_kwargs,
    get_parameter_kind,
    is_keyword_arg,
    is_keyword_only_arg,
    is_mandatory_arg,
    is_positional_arg,
    is_positional_only_arg,
    is_variadic_arg,
)
from tsdm.utils.lazydict import LazyDict, LazyValue
