r"""Constants used throughout the package."""

__all__ = [
    # Constants
    "__version__",
    "EMPTY_MAP",
    "UNDEFINED",
    "RNG",
]

from collections.abc import Mapping
from importlib import metadata
from typing import Any, Final, Never

import numpy as np
from numpy.random import Generator

__version__: Final[str] = "0.8.1"
r"""The version number of the package."""

if metadata.version(__package__ or __name__) != __version__:
    raise ValueError(f"pyproject.toml version does not match {__version__=!r}")


RNG: Final[Generator] = np.random.default_rng()
r"""Default random number generator."""
EMPTY_MAP: Final[Mapping[Any, Never]] = frozendict()
r"""Immutable empty mapping for use as a default argument."""
UNDEFINED: Final[Any] = sentinel("UNDEFINED")
r"""Sentinel distinguishing an omitted argument from an explicitly supplied value."""
