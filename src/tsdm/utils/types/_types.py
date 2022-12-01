r"""Generic types for type hints etc."""

# from __future__ import annotations

__all__ = [
    # Type Variables
    "AliasVar",
    "AnyTypeVar",
    "ClassVar",
    "DtypeVar",
    "IDVar",
    "KeyVar",
    "Key_co",
    "ModuleVar",
    "NestedKeyVar",
    "NestedKey_co",
    "ObjectVar",
    "PandasVar",
    "PathVar",
    "ReturnVar",
    "T",
    "T_co",
    "T_contra",
    "TensorVar",
    "TorchModuleVar",
    "ValueVar",
    # Type Aliases
    "Args",
    "KWArgs",
    "Nested",
    "PandasObject",
    "PathType",
    # ParamSpec
    "Parameters",
]

import os
from collections.abc import Collection, Hashable, Mapping, Sequence
from pathlib import Path
from types import GenericAlias, ModuleType
from typing import Any, ParamSpec, TypeAlias, TypeVar

from numpy import ndarray
from pandas import DataFrame, Index, MultiIndex, Series
from torch import Tensor, nn

Parameters = ParamSpec("Parameters")
r"""TypeVar for decorated function inputs values."""

# region TypeVars

T = TypeVar("T")
r"""Generic type variable."""
T_co = TypeVar("T_co", covariant=True)
r"""Generic type variable for covariant types."""
T_contra = TypeVar("T_contra", contravariant=True)
r"""Generic type variable for contravariant types."""

AnyTypeVar = TypeVar("AnyTypeVar")
r"""Type Variable arbitrary types.."""

AliasVar = TypeVar("AliasVar", bound=GenericAlias)
r"""Type Variable `TypeAlias`."""

ClassVar = TypeVar("ClassVar", bound=type)
r"""Type Variable for `type`."""

DtypeVar = TypeVar("DtypeVar")
r"""Type Variable for `Dtype`."""

KeyVar = TypeVar("KeyVar", bound=Hashable)
r"""Type Variable for `Mapping` keys."""

IDVar = TypeVar("IDVar", bound=Hashable)
r"""Alternative type Variable for `Mapping` keys."""

Key_co = TypeVar("Key_co", bound=Hashable, covariant=True)
r"""type Variable for `Mapping` keys  (covariant)."""

NestedKeyVar = TypeVar("NestedKeyVar", bound=Hashable)
r"""Type Variable for `Mapping` keys."""

NestedKey_co = TypeVar("NestedKey_co", bound=Hashable, covariant=True)
r"""Type Variable for `Mapping` keys."""

ModuleVar = TypeVar("ModuleVar", bound=ModuleType)
r"""Type Variable for Modules."""

ObjectVar = TypeVar("ObjectVar", bound=object)
r"""Type Variable for `object`."""

PandasVar = TypeVar("PandasVar", Index, Series, DataFrame)
r"""Type Variable for `pandas` objects."""

PathVar = TypeVar("PathVar", str, Path, os.PathLike[str])
r"""Type Variable for path-like objects."""

ReturnVar = TypeVar("ReturnVar")
r"""Type Variable for return values."""

TensorVar = TypeVar("TensorVar", Tensor, ndarray)
r"""Type Variable for `torch.Tensor` or `numpy.ndarray` objects."""

TorchModuleVar = TypeVar("TorchModuleVar", bound=nn.Module)
r"""Type Variable for `torch.nn.Modules`."""

ValueVar = TypeVar("ValueVar")
r"""Type Variable for `Mapping` values."""

ValueVar_co = TypeVar("ValueVar_co", covariant=True)
r"""Type Variable for `Mapping` values (covariant)."""

# endregion

# region TypeAliases
Args: TypeAlias = tuple[AnyTypeVar, ...]
r"""Type Alias for `*args`."""

KWArgs: TypeAlias = dict[str, AnyTypeVar]
r"""Type Alias for `**kwargs`."""

Nested: TypeAlias = AnyTypeVar | Collection[AnyTypeVar] | Mapping[Any, AnyTypeVar]
r"""Type Alias for nested types (JSON-Like)."""

PandasObject: TypeAlias = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""

PathType: TypeAlias = str | Path  # | os.PathLike[str]
r"""Type Alias for path-like objects."""
# endregion


# region Recursive Type Aliases
JSON: TypeAlias = None | str | int | float | bool | list["JSON"] | dict[str, "JSON"]  # type: ignore[operator]
r"""Type Alias for JSON-Like objects."""

TOML: TypeAlias = None | str | int | float | bool | list["TOML"] | dict[str, "TOML"]  # type: ignore[operator]

NestedDict: TypeAlias = dict[KeyVar, ValueVar | dict[KeyVar, "NestedDict"]]
r"""Generic Type Alias for nested `dict`."""

NestedMapping: TypeAlias = Mapping[KeyVar, ValueVar | Mapping[KeyVar, "NestedMapping"]]
r"""Generic Type Alias for nested `Mapping`."""

NestedList: TypeAlias = list[ValueVar | list["NestedList"]]
r"""GenericType Alias for nested `list`."""

NestedSequence: TypeAlias = Sequence[ValueVar | Sequence[ValueVar]]
r"""Generic Type Alias for nested `Sequence`."""

NestedSet: TypeAlias = set[ValueVar | set["NestedSet"]]
r"""Generic Type Alias for nested `set`."""

NestedTuple: TypeAlias = tuple[ValueVar | tuple["NestedTuple", ...], ...]
r"""Generic Type Alias for nested `tuple`."""
# endregion
