r"""Generic types for type hints etc."""

__all__ = [
    # Custom Type Aliases
    "alias_var",
    "dtype_var",
    "pandas_var",
    "tensor_var",
    "torch_module_var",
    # Builtin Type Variables
    "class_var",
    "class_co",
    "class_contra",
    "dict_var",
    "dict_co",
    "dict_contra",
    "module_var",
    "module_co",
    "module_contra",
    "list_var",
    "list_co",
    "list_contra",
    "object_var",
    "object_co",
    "object_contra",
    "path_var",
    "path_co",
    "path_contra",
    "set_var",
    "set_co",
    "set_contra",
    "str_var",
    "str_co",
    "str_contra",
    "tuple_var",
    "tuple_co",
    "tuple_contra",
    # Generic Type Variables
    "any_var",
    "any_co",
    "any_contra",
    "any2_var",
    "any2_co",
    "any2_contra",
    "input_var_contra",
    "key_var",
    "key_co",
    "key_contra",
    "key2_var",
    "key2_co",
    "key2_contra",
    "nested_key_var",
    "nested_key_co",
    "nested_key_contra",
    "parameter_spec",
    "return_var_co",
    "value_var",
    "value_co",
    "value_contra",
]

from collections.abc import Hashable
from pathlib import Path
from types import GenericAlias, ModuleType
from typing import ParamSpec, TypeVar

from numpy import ndarray
from pandas import DataFrame, Index, MultiIndex, Series
from torch import Tensor, nn

# region Custom Type Variables ---------------------------------------------------------
alias_var = TypeVar("alias_var", bound=GenericAlias)
r"""Type Variable `TypeAlias`."""
dtype_var = TypeVar("dtype_var")
r"""Type Variable for `Dtype`."""
pandas_var = TypeVar("pandas_var", Index, Series, DataFrame, MultiIndex)
r"""Type Variable for `pandas` objects."""
tensor_var = TypeVar("tensor_var", Tensor, ndarray)
r"""Type Variable for `torch.Tensor` or `numpy.ndarray` objects."""
torch_module_var = TypeVar("torch_module_var", bound=nn.Module)
r"""Type Variable for `torch.nn.Modules`."""
# endregion Custom Type Variables ------------------------------------------------------


# Type variables of Built-in Types
# region Builtins ----------------------------------------------------------------------
class_var = TypeVar("class_var", bound=type)
r"""Type Variable for `type`."""
class_co = TypeVar("class_co", bound=type, covariant=True)
r"""Type Variable for `type`."""
class_contra = TypeVar("class_contra", bound=type, contravariant=True)
r"""Type Variable for `type`."""

dict_var = TypeVar("dict_var", bound=dict)
r"""Type Variable for `dict`."""
dict_co = TypeVar("dict_co", bound=dict, covariant=True)
r"""Covariant type variable for `dict`."""
dict_contra = TypeVar("dict_contra", bound=dict, contravariant=True)
r"""Contravariant type variable for `dict`."""

module_var = TypeVar("module_var", bound=ModuleType)
r"""Type Variable for `ModuleType`."""
module_co = TypeVar("module_co", bound=ModuleType, covariant=True)
r"""Covariant type variable for `ModuleType`."""
module_contra = TypeVar("module_contra", bound=ModuleType, contravariant=True)
r"""Contravariant type variable for `ModuleType`."""

list_var = TypeVar("list_var", bound=list)
r"""Type Variable for `list`."""
list_co = TypeVar("list_co", bound=list, covariant=True)
r"""Covariant type variable for `list`."""
list_contra = TypeVar("list_contra", bound=list, contravariant=True)
r"""Contravariant type variable for `list`."""

object_var = TypeVar("object_var", bound=object)
r"""Type Variable for `object`."""
object_co = TypeVar("object_co", bound=object, covariant=True)
r"""Covariant type variable for `object`."""
object_contra = TypeVar("object_contra", bound=object, contravariant=True)
r"""Contravariant type variable for `object`."""

path_var = TypeVar("path_var", bound=Path)
r"""Type Variable for `Path`."""
path_co = TypeVar("path_co", bound=Path, covariant=True)
r"""Covariant type variable for `Path`."""
path_contra = TypeVar("path_contra", bound=Path, contravariant=True)
r"""Contravariant type variable for `Path`."""

set_var = TypeVar("set_var", bound=set)
r"""Type Variable for `set`."""
set_co = TypeVar("set_co", bound=set, covariant=True)
r"""Covariant type variable for `set`."""
set_contra = TypeVar("set_contra", bound=set, contravariant=True)
r"""Contravariant type variable for `set`."""

str_var = TypeVar("str_var", bound=str)
r"""Type Variable for `str`."""
str_co = TypeVar("str_co", bound=str, covariant=True)
r"""Covariant type variable for `str`."""
str_contra = TypeVar("str_contra", bound=str, contravariant=True)
r"""Contravariant type variable for `str`."""

tuple_var = TypeVar("tuple_var", bound=tuple)
r"""Type Variable for `tuple`."""
tuple_co = TypeVar("tuple_co", bound=tuple, covariant=True)
r"""Covariant type variable for `tuple`."""
tuple_contra = TypeVar("tuple_contra", bound=tuple, contravariant=True)
r"""Contravariant type variable for `tuple`."""
# endregion Builtins -------------------------------------------------------------------


# Generic purpose type variables
# region Generic Type Variables --------------------------------------------------------
any_var = TypeVar("any_var")
r"""Generic type variable."""
any_co = TypeVar("any_co", covariant=True)
r"""Generic covariant type variable."""
any_contra = TypeVar("any_contra", contravariant=True)
r"""Generic contravariant type variable."""

any2_var = TypeVar("any2_var")
r"""Generic type variable."""
any2_co = TypeVar("any2_co", covariant=True)
r"""Generic covariant type variable."""
any2_contra = TypeVar("any2_contra", contravariant=True)
r"""Generic contravariant type variable."""

# inputs are always contravariant!
input_var_contra = TypeVar("input_var_contra", contravariant=True)
r"""Generic type variable for inputs. Always contravariant."""

key_var = TypeVar("key_var", bound=Hashable)
r"""Type Variable for `Mapping` keys."""
key_co = TypeVar("key_co", bound=Hashable, covariant=True)
r"""Covariant type Variable for `Mapping` keys."""
key_contra = TypeVar("key_contra", bound=Hashable, contravariant=True)
r""" Contravariant type Variable for `Mapping` keys."""

key2_var = TypeVar("key2_var", bound=Hashable)
r"""Type Variable for `Mapping` keys."""
key2_co = TypeVar("key2_co", bound=Hashable, covariant=True)
r"""Covariant type Variable for `Mapping` keys."""
key2_contra = TypeVar("key2_contra", bound=Hashable, contravariant=True)
r""" Contravariant type Variable for `Mapping` keys."""

nested_key_var = TypeVar("nested_key_var", bound=Hashable)
r"""Type Variable for `Mapping` keys."""
nested_key_co = TypeVar("nested_key_co", bound=Hashable, covariant=True)
r"""Covariant type variable for `Mapping` keys."""
nested_key_contra = TypeVar("nested_key_contra", bound=Hashable, contravariant=True)
r"""Contravariant type variable for `Mapping` keys."""

parameter_spec = ParamSpec("parameter_spec")
r"""TypeVar for decorated function inputs values."""

# outputs/returns are always covariant!
return_var_co = TypeVar("return_var_co", covariant=True)
r"""Type Variable for return values. Always covariant."""

value_var = TypeVar("value_var")
r"""Type Variable for `Mapping` values."""
value_co = TypeVar("value_co", covariant=True)
r"""Covariant type variable for `Mapping` values."""
value_contra = TypeVar("value_contra", contravariant=True)
r"""Contravariant type variable for `Mapping` values."""
# endregion Generic Type Variables -----------------------------------------------------
