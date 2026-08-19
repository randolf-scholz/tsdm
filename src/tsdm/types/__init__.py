r"""Generic types for type hints, etc."""

__all__ = [
    # Submodules
    "abc",
    "aliases",
    "callbacks",
    "dataclass",
    "namedtuple",
    # Aliases
    "FilePath",
    "FileStream",
    "DictArg",
    "Axis",
    "Size",
    # Protocols
    "SupportsBool",
    "SupportsGetItem",
    "SupportsKeysAndGetItem",
    "SupportsLenAndGetItem",
    "SupportsSlicing",
    # Comparison operations
    "SupportsEquality",
    "SupportsComparison",
    # Mixins
    "SupportsArray",
    "SupportsArrayUfunc",
    "SupportsDataFrame",
    "SupportsDevice",
    "SupportsDtype",
    "SupportsItem",
    "SupportsNdim",
    "SupportsRound",
    "SupportsShape",
    # Special
    "Dataclass",
    "NTuple",
    "is_dataclass",
    "is_namedtuple",
]

from . import abc, aliases, callbacks, dataclass, namedtuple
from .aliases import Axis, DictArg, FilePath, FileStream, Size
from .dataclass import Dataclass, is_dataclass
from .namedtuple import NTuple, is_namedtuple
from .protocols import (
    SupportsArray,
    SupportsArrayUfunc,
    SupportsBool,
    SupportsComparison,
    SupportsDataFrame,
    SupportsDevice,
    SupportsDtype,
    SupportsEquality,
    SupportsGetItem,
    SupportsItem,
    SupportsKeysAndGetItem,
    SupportsLenAndGetItem,
    SupportsNdim,
    SupportsRound,
    SupportsShape,
    SupportsSlicing,
)
