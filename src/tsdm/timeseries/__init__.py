r"""Utilities for time series."""

__all__ = [
    # submodules:
    "abstract",
    "pandas",
    "polars",
]

from . import abstract, pandas, polars
from .abstract import *  # ruff: ignore[F403]

__all__ += abstract.__all__
