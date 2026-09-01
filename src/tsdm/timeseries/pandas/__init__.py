r"""Pandas-backed time-series datasets and prediction samples.

This backend represents dataset tables with pandas objects, preserving their
index-based time and series identifiers. It exports wrappers for single and
multiple time series together with factories for split prediction samples.
"""

__all__ = ["datasets", "samples"]

from . import datasets, samples
from .datasets import *  # ruff: ignore[F403]
from .samples import *  # ruff: ignore[F403]

__all__ += datasets.__all__
__all__ += samples.__all__
