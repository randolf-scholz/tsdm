r"""Generic types for type hints etc."""

from __future__ import annotations

__all__ = [
    # Type Variables
    "Self",
    "ClassType",
    "KeyType",
    "ObjectType",
    "ReturnType",
    "ValueType",
    "nnModuleType",
    "Parameters",
    "Type",
    # Generic Types
    "Nested",
    # Static Types
    "PathType",
    "PandasObject",
    # Time Types
    "PandasVar",
    "TensorVar",
]

import os
import sys
import typing
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from typing import Any, TypeVar, Union

from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from torch import Tensor, nn

if sys.version_info >= (3, 10):
    ParamSpec = getattr(typing, "ParamSpec")  # noqa: B009
else:
    ParamSpec = typing.TypeVar

ArgsType = TypeVar("ArgsType")
r"""TypeVar for `Mapping` values."""

KeyType = TypeVar("KeyType", bound=Hashable)
r"""TypeVar for `Mapping` keys."""

ValueType = TypeVar("ValueType")
r"""TypeVar for `Mapping` values."""

Type = TypeVar("Type")
r"""TypeVar for `Mapping` values."""

Self = TypeVar("Self")
r"""TypeVar for for self reference."""  # FIXME: PEP673 @ python3.11

ClassType = TypeVar("ClassType", bound=type)
r"""Generic type hint."""

Parameters = ParamSpec("Parameters")
r"""TypeVar for decorated function inputs values."""

ObjectType = TypeVar("ObjectType", bound=object)
r"""Generic type hint for instances."""

ReturnType = TypeVar("ReturnType")
r"""Generic type hint for return values."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for nn.Modules."""

# PathType = TypeVar("PathType", bound=Union[str, Path])
# r"""Type for path-like objects."""

PathType = Union[str, Path, os.PathLike[str]]
r"""Type for path-like objects."""

PathVar = TypeVar("PathVar", str, Path, os.PathLike[str])
r"""TypeVar for path-like objects."""

Nested = Union[
    Type,
    Collection[Type],
    Mapping[Any, Type],
]
r"""Type for nested types (JSON-Like)."""

PandasObject = Union[Index, Series, DataFrame]
r"""Type Hint for pandas objects."""

PandasVar = TypeVar("PandasVar", Index, Series, DataFrame)
r"""Type Hint for pandas objects."""

TensorVar = TypeVar("TensorVar", Tensor, NDArray)
