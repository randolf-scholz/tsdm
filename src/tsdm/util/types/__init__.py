r"""Generic types for type hints etc."""

__all__ = [
    # Submodules
    "abc",
    "protocols",
    # Type Variables
    "Self",
    "ClassType",
    "KeyType",
    "ObjectType",
    "ReturnType",
    "ValueType",
    "Type",
    "nnModuleType",
    # Generic Types
    "Nested",
    # Static Types
    "PathType",
    "PandasObject",
]

import logging

from tsdm.util.types import abc, protocols
from tsdm.util.types._types import (
    ClassType,
    KeyType,
    Nested,
    ObjectType,
    PandasObject,
    PathType,
    ReturnType,
    Self,
    Type,
    ValueType,
    nnModuleType,
)

__logger__ = logging.getLogger(__name__)
