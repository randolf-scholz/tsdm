r"""Time-series dataset containers and prediction-sample representations.

The subpackage defines backend-neutral contracts in :mod:`.abstract` and
provides compatible implementations for pandas, Polars, and PyTorch. These
implementations wrap dataset tables as single series or collections of series,
and represent samples used by time-series prediction tasks.
"""

__all__ = [
    # submodules:
    "abstract",
    "pandas",
    "polars",
]

from . import abstract, pandas, polars
from .abstract import *  # ruff: ignore[F403]

__all__ += abstract.__all__
