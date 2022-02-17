r"""Generic types for type hints etc."""

__all__ = [
    # Submodules
    "abc",
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

from tsdm.util.types import abc
from tsdm.util.types._types import (
    ClassType,
    KeyType,
    LookupTable,
    NestedType,
    NullableNestedType,
    ObjectType,
    PathType,
    ReturnType,
    Type,
    ValueType,
    nnModuleType,
)

__logger__ = logging.getLogger(__name__)
