r"""Generic types for type hints, etc."""

__all__ = [
    # Submodules
    "abc",
    "aliases",
    "callbacks",
    "dataclass",
    "namedtuple",
    "protocols",
    "nested",
    # Special
    "Dataclass",
    "NTuple",
    "is_dataclass",
    "is_namedtuple",
]

from . import abc, aliases, callbacks, dataclass, namedtuple, nested, protocols
from .aliases import *  # ruff: ignore[F403]
from .dataclass import Dataclass, is_dataclass
from .namedtuple import NTuple, is_namedtuple
from .protocols import *  # ruff: ignore[F403]

__all__ += aliases.__all__
__all__ += protocols.__all__
