r"""Generic types for type hints etc."""

__all__ = [
    # Type Variables
    "Self",
    "ClassType",
    "KeyType",
    "ObjectType",
    "ReturnType",
    "ValueType",
    "nnModuleType",
    "Type",
    # Generic Types
    "Nested",
    # Static Types
    "PathType",
]

import logging
import os
from collections.abc import Collection, Hashable, Mapping
from pathlib import Path
from typing import Any, TypeVar, Union

from torch import nn

__logger__ = logging.getLogger(__name__)


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

ObjectType = TypeVar("ObjectType", bound=object)
r"""Generic type hint for instances."""

ReturnType = TypeVar("ReturnType")
r"""Generic type hint for return values."""

nnModuleType = TypeVar("nnModuleType", bound=nn.Module)
r"""Type Variable for nn.Modules."""

# PathType = TypeVar("PathType", bound=Union[str, Path])
# r"""Type for path-like objects."""

PathType = Union[str, Path, os.PathLike[str]]

Nested = Union[
    Type,
    Collection[Type],
    Mapping[Any, Type],
]
r"""Type for nested types (JSON-Like)."""
