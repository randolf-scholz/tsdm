r"""Provides utility functions."""

__all__ = [
    # Sub-Packages
    "decorators",
    # Sub-Modules
    "remote",
    "system",
    # Constants
    # Classes
    "LazyDict",
    "LazyValue",
    # utils
    "axes_to_tuple",
    "deep_dict_update",
    "deep_kval_update",
    "dims_to_list",
    "flatten_dict",
    "flatten_nested",
    "initialize_from_config",
    "joint_keys",
    "last",
    "now",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
    "paths_exists",
    "prepend_path",
    "repackage_zip",
    "replace",
    "round_relative",
    "size_to_tuple",
    "transpose_list_of_dicts",
    "unflatten_dict",
    "unpack_maybewrapped",
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

from tsdm.utils import decorators, remote, system
from tsdm.utils._utils import (
    axes_to_tuple,
    deep_dict_update,
    deep_kval_update,
    dims_to_list,
    flatten_dict,
    flatten_nested,
    initialize_from_config,
    joint_keys,
    last,
    now,
    pairwise_disjoint,
    pairwise_disjoint_masks,
    paths_exists,
    prepend_path,
    repackage_zip,
    replace,
    round_relative,
    size_to_tuple,
    transpose_list_of_dicts,
    unflatten_dict,
    unpack_maybewrapped,
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
