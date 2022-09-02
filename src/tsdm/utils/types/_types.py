r"""Generic types for type hints etc."""

from __future__ import annotations

__all__ = [
    # Type Variables
    "ArgsType",
    "ClassType",
    "KeyType",
    "nnModuleType",
    "ObjectType",
    "PandasVar",
    "PathVar",
    "ReturnType",
    "Self",
    "TensorVar",
    "Type",
    "ValueType",
    # Type Aliases
    "PathType",
    "Nested",
    "PandasObject",
    # ParamSpec
    "Parameters",
]

import os
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from typing import Any, ParamSpec, TypeAlias, TypeVar

from numpy import ndarray
from pandas import DataFrame, Index, Series
from torch import Tensor, nn

Parameters = ParamSpec("Parameters")
r"""TypeVar for decorated function inputs values."""

# region TypeVars

ArgsType = TypeVar("ArgsType")
r"""TypeVar for `Mapping` values."""

ClassType = TypeVar("ClassType", bound=type)
r"""Generic type hint."""

KeyType = TypeVar("KeyType", bound=Hashable)
r"""TypeVar for `Mapping` keys."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for `torch.nn.Modules`."""

ObjectType = TypeVar("ObjectType", bound=object)
r"""Generic type hint for instances."""

PandasVar = TypeVar("PandasVar", Index, Series, DataFrame)
r"""Type Variable for `pandas` objects."""

PathVar = TypeVar("PathVar", str, Path, os.PathLike[str])
r"""TypeVar for path-like objects."""

ReturnType = TypeVar("ReturnType")
r"""Generic type hint for return values."""

Self = TypeVar("Self")
r"""TypeVar for for self reference."""  # FIXME: PEP673 @ python3.11

TensorVar = TypeVar("TensorVar", Tensor, ndarray)
r"""Type Variable for `torch.Tensor` or `numpy.ndarray` objects."""

Type = TypeVar("Type")
r"""TypeVar for `Mapping` values."""
ValueType = TypeVar("ValueType")
r"""TypeVar for `Mapping` values."""

# endregion

# region TypeAliases

PathType: TypeAlias = str | Path | os.PathLike[str]
r"""Type for path-like objects."""

Nested: TypeAlias = Type | Collection[Type] | Mapping[Any, Type]
r"""Type for nested types (JSON-Like)."""

PandasObject: TypeAlias = Index | Series | DataFrame
r"""Type Hint for `pandas` objects."""

# endregion
