r"""Generic types for type hints etc."""

__all__ = [
    # Type Variables
    "Type",
    "ClassType",
    "KeyType",
    "ObjectType",
    "ReturnType",
    "ValueType",
    "nnModuleType",
    # Generic Types
    "LookupTable",
    "NestedType",
    "NullableNestedType",
    # Static Types
    "PathType",
]

import logging
import os
from collections.abc import Collection, Mapping
from typing import Any, TypeVar, Union

from torch import nn

__logger__ = logging.getLogger(__name__)

KeyType = TypeVar("KeyType")
r"""TypeVar for ``Mapping`` keys."""

ValueType = TypeVar("ValueType")
r"""TypeVar for ``Mapping`` values."""

Type = TypeVar("Type")
r"""Generic type hint"""

ClassType = TypeVar("ClassType", bound=type)
r"""Generic type hint"""

ObjectType = TypeVar("ObjectType", bound=object)
r"""Generic type hint for instances."""

ReturnType = TypeVar("ReturnType")
r"""Generic type hint for return values."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for nn.Modules."""

_NestedType = TypeVar("_NestedType")
r"""Type variable for nested type variables."""

LookupTable = dict[str, ObjectType]
r"""Table of objects."""

NestedType = Union[
    _NestedType,
    Collection[_NestedType],
    Mapping[Any, Union[_NestedType, Collection[_NestedType]]],
]
r"""Type for nested type variables."""

NullableNestedType = Union[
    None,
    _NestedType,
    Collection[_NestedType],
    Mapping[Any, Union[None, _NestedType, Collection[_NestedType]]],
]
r"""Type for nullable nested type variables."""

PathType = Union[str, os.PathLike[str]]
r"""Type for path-like objects."""
