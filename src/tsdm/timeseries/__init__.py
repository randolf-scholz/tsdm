r"""Utilities for time series."""

__all__ = [
    # submodules:
    "base",
    "pandas",
    "polars",
    # ABCs & Protocols
    "TimeSeriesCollection",
    "TimeSeries",
    # classes
    "PandasTS",
    "PandasTSC",
    "PolarsTS",
    "PolarsTSC",
    # Functions
]

from . import base, pandas, polars
from .base import TimeSeries, TimeSeriesCollection
from .pandas import PandasTS, PandasTSC
from .polars import PolarsTS, PolarsTSC
