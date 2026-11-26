r"""Type aliases, protocols, and runtime type checks for TSDM."""

__all__ = [
    # Submodules
    "aliases",
    "dataclass",
    "datasets",
    "namedtuple",
    "protocols",
    # Special
    "DataclassType",
    "NamedTupleType",
    "is_dataclass",
    "is_namedtuple",
]

from . import aliases, dataclass, datasets, namedtuple, protocols
from .aliases import *  # ruff: ignore[F403]
from .dataclass import DataclassType, is_dataclass
from .datasets import *  # ruff: ignore[F403]
from .namedtuple import NamedTupleType, is_namedtuple
from .protocols import *  # ruff: ignore[F403]

__all__ += aliases.__all__
__all__ += protocols.__all__
__all__ += datasets.__all__
