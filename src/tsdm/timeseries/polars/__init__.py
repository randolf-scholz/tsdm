r"""Polars-backed time-series datasets and sample interfaces.

This backend provides table containers based on Polars DataFrames, using an
explicit timestamp column because Polars has no row index. Its public surface
also reserves a home for Polars-native prediction-sample implementations.
"""

__all__ = ["datasets", "samples"]

from . import datasets, samples
from .datasets import *  # ruff: ignore[F403]
from .samples import *  # ruff: ignore[F403]

__all__ += datasets.__all__
__all__ += samples.__all__
