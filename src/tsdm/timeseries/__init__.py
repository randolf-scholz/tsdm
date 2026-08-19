r"""Utilities for time series."""

__all__ = [
    # submodules:
    "base",
    "pandas",
    "polars",
    "sample_generators",
    "util",
    # ABCs & Protocols
    "TimeSeriesCollection",
    "TimeSeries",
    "TimeSeriesSampleGenerator",
    "FixedSliceSampleGenerator",
    # classes
    "PandasTS",
    "PandasTSC",
    "PolarsTS",
    "PolarsTSC",
    "Inputs",
    "Targets",
    "Sample",
    "PlainSample",
    "TimeSeriesSample",
    "PaddedBatch",
    # Functions
    "collate_timeseries",
]

from . import base, pandas, polars, sample_generators, util
from .base import TimeSeries, TimeSeriesCollection
from .pandas import PandasTS, PandasTSC
from .polars import PolarsTS, PolarsTSC
from .sample_generators import (
    FixedSliceSampleGenerator,
    Inputs,
    PlainSample,
    Sample,
    Targets,
    TimeSeriesSampleGenerator,
)
from .util import PaddedBatch, TimeSeriesSample, collate_timeseries
