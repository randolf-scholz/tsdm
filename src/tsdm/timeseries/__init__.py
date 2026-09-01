r"""Utilities for time series."""

__all__ = [
    # submodules:
    "abstract",
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

from . import abstract, pandas, polars
from .abstract import TimeSeries, TimeSeriesCollection
from .pandas import PandasTS, PandasTSC
from .polars import PolarsTS, PolarsTSC
