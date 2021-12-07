r"""Generic types for type hints etc."""

__all__ = [
    # Submodules
    "abc",
    # Type Variables
    "ObjectType",
    "ReturnType",
    # Types
    "LookupTable",
    "PathType",
    "NullableNestedType",
    "NestedType",
    "KeyType",
]

import logging

from tsdm.util.types import abc
from tsdm.util.types._types import (
    KeyType,
    LookupTable,
    NestedType,
    NullableNestedType,
    ObjectType,
    PathType,
    ReturnType,
)

__logger__ = logging.getLogger(__name__)
