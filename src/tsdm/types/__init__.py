r"""Generic types for type hints, etc."""

__all__ = [
    # Submodules
    "abc",
    "aliases",
    "callbacks",
    "dataclass",
    "utils",
    "namedtuple",
    # Protocols
    "Orderable",
    "SupportsBool",
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsLenAndGetItem",
    "SupportsSlicing",
]

from . import abc, aliases, callbacks, dataclass, namedtuple, utils
from .protocols import (
    Orderable,
    SupportsBool,
    SupportsGetItem,
    SupportsKeysAndGetItem,
    SupportsLenAndGetItem,
    SupportsSlicing,
)
