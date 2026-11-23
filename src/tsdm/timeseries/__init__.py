r"""Time-series dataset containers and prediction-sample representations.

The subpackage defines backend-neutral contracts in `abstract` and
provides compatible implementations for `pandas`, `polars`, and `torch`. These
implementations wrap dataset tables as single series or collections of series,
and represent samples used by time-series prediction tasks.
"""

__all__ = [
    # submodules:
    "abstract",
    "pandas",
    "polars",
]

from . import abstract
from .abstract import *  # ruff: ignore[F403]
lazy from . import pandas, polars

__all__ += abstract.__all__
