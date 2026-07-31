r"""Constants used throughout the package."""

__all__ = [
    # ENUMS
    "FLOAT",
    # Constants
    "EMPTY_MAP",
    "EMPTY_SET",
    "UNDEFINED",
    "RNG",
]

import math
from collections.abc import Mapping, Set as AbstractSet
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Never

import numpy as np
from numpy.random import Generator

RNG: Final[Generator] = np.random.default_rng()
r"""Default random number generator."""
EMPTY_MAP: Final[Mapping[Any, Never]] = MappingProxyType({})  # FIXME: PEP 603
r"""Constant: Immutable empty `Mapping`, use as default in function signatures."""
EMPTY_SET: Final[AbstractSet[Any]] = frozenset()
r"""Constant: Immutable empty `Set`, use as default in function signatures."""
UNDEFINED: Final[Any] = object()
r"""CONST: Default value for optional arguments."""


class FLOAT(float, Enum):
    r"""Enum: Common floating point values."""

    ZERO = 0.0
    ONE = 1.0
    INF = math.inf
    NAN = math.nan

    E = math.e
    PI = math.pi
    ROOT_2 = math.sqrt(2)
    ROOT_2PI = math.sqrt(2 * math.pi)
    ROOT_3 = math.sqrt(3)
