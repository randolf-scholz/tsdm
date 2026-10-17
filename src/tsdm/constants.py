r"""Constants used throughout the package."""

__all__ = [
    # Constants
    "EMPTY_MAP",
    "UNDEFINED",
    "RNG",
]

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, Never

import numpy as np
from numpy.random import Generator

RNG: Final[Generator] = np.random.default_rng()
r"""Default random number generator."""
EMPTY_MAP: Final[Mapping[Any, Never]] = MappingProxyType({})  # FIXME: PEP 603
r"""Constant: Immutable empty `Mapping`, use as default in function signatures."""
UNDEFINED: Final[Any] = object()
r"""CONST: Default value for optional arguments."""
