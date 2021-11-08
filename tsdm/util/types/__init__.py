r"""Generic types for type hints etc."""

from __future__ import annotations

__all__ = [
    # Type Variables
    "ObjectType",
    "ReturnType",
    # Types
    "LookupTable",
]

import logging

from tsdm.util.types._types import LookupTable, ObjectType, ReturnType

__logger__ = logging.getLogger(__name__)
