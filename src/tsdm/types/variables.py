r"""Generic types for type hints etc."""

__all__ = [
    # Custom Type Aliases
    "AliasVar",
    "DtypeVar",
    "PandasVar",
    "TensorVar",
    "TorchModuleVar",
    # Builtin Type Variables
    "ClassVar",
    "Class_co",
    "Class_contra",
    "DictVar",
    "Dict_co",
    "Dict_contra",
    "ModuleVar",
    "Module_co",
    "Module_contra",
    "ListVar",
    "List_co",
    "List_contra",
    "ObjectVar",
    "Object_co",
    "Object_contra",
    "PathVar",
    "Path_co",
    "Path_contra",
    "SetVar",
    "Set_co",
    "Set_contra",
    "TupleVar",
    "Tuple_co",
    "Tuple_contra",
    # Generic Type Variables
    "AnyVar",
    "Any_co",
    "Any_contra",
    "InputVar",
    "KeyVar",
    "Key_co",
    "Key_contra",
    "NestedKeyVar",
    "NestedKey_co",
    "NestedKey_contra",
    "ParameterVar",
    "ReturnVar",
    "ValueVar",
    "ValueVar_co",
    "ValueVar_contra",
]

from collections.abc import Hashable
from pathlib import Path
from types import GenericAlias, ModuleType
from typing import ParamSpec, TypeVar

from numpy import ndarray
from pandas import DataFrame, Index, MultiIndex, Series
from torch import Tensor, nn

# region Custom Type Variables ---------------------------------------------------------
AliasVar = TypeVar("AliasVar", bound=GenericAlias)
r"""Type Variable `TypeAlias`."""
DtypeVar = TypeVar("DtypeVar")
r"""Type Variable for `Dtype`."""
PandasVar = TypeVar("PandasVar", Index, Series, DataFrame, MultiIndex)
r"""Type Variable for `pandas` objects."""
TensorVar = TypeVar("TensorVar", Tensor, ndarray)
r"""Type Variable for `torch.Tensor` or `numpy.ndarray` objects."""
TorchModuleVar = TypeVar("TorchModuleVar", bound=nn.Module)
r"""Type Variable for `torch.nn.Modules`."""
# endregion Custom Type Variables ------------------------------------------------------


# Type variables of Built-in Types
# region Builtins ----------------------------------------------------------------------
ClassVar = TypeVar("ClassVar", bound=type)
r"""Type Variable for `type`."""
Class_co = TypeVar("Class_co", bound=type, covariant=True)
r"""Type Variable for `type`."""
Class_contra = TypeVar("Class_contra", bound=type, contravariant=True)
r"""Type Variable for `type`."""

DictVar = TypeVar("DictVar", bound=dict)
r"""Type Variable for `dict`."""
Dict_co = TypeVar("Dict_co", bound=dict, covariant=True)
r"""Covariant type variable for `dict`."""
Dict_contra = TypeVar("Dict_contra", bound=dict, contravariant=True)
r"""Contravariant type variable for `dict`."""

ModuleVar = TypeVar("ModuleVar", bound=ModuleType)
r"""Type Variable for `ModuleType`."""
Module_co = TypeVar("Module_co", bound=ModuleType, covariant=True)
r"""Covariant type variable for `ModuleType`."""
Module_contra = TypeVar("Module_contra", bound=ModuleType, contravariant=True)
r"""Contravariant type variable for `ModuleType`."""

ListVar = TypeVar("ListVar", bound=list)
r"""Type Variable for `list`."""
List_co = TypeVar("List_co", bound=list, covariant=True)
r"""Covariant type variable for `list`."""
List_contra = TypeVar("List_contra", bound=list, contravariant=True)
r"""Contravariant type variable for `list`."""

ObjectVar = TypeVar("ObjectVar", bound=object)
r"""Type Variable for `object`."""
Object_co = TypeVar("Object_co", bound=object, covariant=True)
r"""Covariant type variable for `object`."""
Object_contra = TypeVar("Object_contra", bound=object, contravariant=True)
r"""Contravariant type variable for `object`."""

PathVar = TypeVar("PathVar", bound=Path)
r"""Type Variable for `Path`."""
Path_co = TypeVar("Path_co", bound=Path, covariant=True)
r"""Covariant type variable for `Path`."""
Path_contra = TypeVar("Path_contra", bound=Path, contravariant=True)
r"""Contravariant type variable for `Path`."""

SetVar = TypeVar("SetVar", bound=set)
r"""Type Variable for `set`."""
Set_co = TypeVar("Set_co", bound=set, covariant=True)
r"""Covariant type variable for `set`."""
Set_contra = TypeVar("Set_contra", bound=set, contravariant=True)
r"""Contravariant type variable for `set`."""

TupleVar = TypeVar("TupleVar", bound=tuple)
r"""Type Variable for `tuple`."""
Tuple_co = TypeVar("Tuple_co", bound=tuple, covariant=True)
r"""Covariant type variable for `tuple`."""
Tuple_contra = TypeVar("Tuple_contra", bound=tuple, contravariant=True)
r"""Contravariant type variable for `tuple`."""
# endregion Builtins -------------------------------------------------------------------


# Generic purpose type variables
# region Generic Type Variables --------------------------------------------------------
AnyVar = TypeVar("AnyVar")
r"""Generic type variable."""
Any_co = TypeVar("Any_co", covariant=True)
r"""Generic covariant type variable."""
Any_contra = TypeVar("Any_contra", contravariant=True)
r"""Generic contravariant type variable."""

InputVar = TypeVar("InputVar", contravariant=True)
r"""Generic type variable for inputs. Always contravariant."""

KeyVar = TypeVar("KeyVar", bound=Hashable)
r"""Type Variable for `Mapping` keys."""
Key_co = TypeVar("Key_co", bound=Hashable, covariant=True)
r"""Covariant type Variable for `Mapping` keys."""
Key_contra = TypeVar("Key_contra", bound=Hashable, contravariant=True)
r""" Contravariant type Variable for `Mapping` keys."""

NestedKeyVar = TypeVar("NestedKeyVar", bound=Hashable)
r"""Type Variable for `Mapping` keys."""
NestedKey_co = TypeVar("NestedKey_co", bound=Hashable, covariant=True)
r"""Covariant type variable for `Mapping` keys."""
NestedKey_contra = TypeVar("NestedKey_contra", bound=Hashable, contravariant=True)
r"""Contravariant type variable for `Mapping` keys."""

ParameterVar = ParamSpec("ParameterVar")
r"""TypeVar for decorated function inputs values."""

ReturnVar = TypeVar("ReturnVar", covariant=True)
r"""Type Variable for return values. Always covariant."""

ValueVar = TypeVar("ValueVar")
r"""Type Variable for `Mapping` values."""
ValueVar_co = TypeVar("ValueVar_co", covariant=True)
r"""Covariant type variable for `Mapping` values."""
ValueVar_contra = TypeVar("ValueVar_contra", contravariant=True)
r"""Contravariant type variable for `Mapping` values."""
# endregion Generic Type Variables -----------------------------------------------------
