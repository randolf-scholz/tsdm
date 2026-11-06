r"""Constants used throughout the package."""

__all__ = [
    # Constants
    "EMPTY_MAP",
    "UNDEFINED",
    "RNG",
]

from collections.abc import Mapping
from typing import Any, Final, Never

import numpy as np
from numpy.random import Generator

RNG: Final[Generator] = np.random.default_rng()
r"""Default random number generator."""
EMPTY_MAP: Final[Mapping[Any, Never]] = frozendict()
r"""Immutable empty mapping for use as a default argument."""
UNDEFINED: Final[Any] = sentinel("UNDEFINED")
r"""Sentinel distinguishing an omitted argument from an explicitly supplied value."""
