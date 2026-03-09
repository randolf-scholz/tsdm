r"""Generic types for type hints, etc."""

__all__ = [
    # Submodules
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

from tsdm.types import aliases, callbacks, dataclass, namedtuple, utils
from tsdm.types._protocols import (
    Orderable,
    SupportsBool,
    SupportsGetItem,
    SupportsKeysAndGetItem,
    SupportsLenAndGetItem,
    SupportsSlicing,
)
