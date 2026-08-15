r"""Utilities for time series."""

__all__ = [
    # submodules:
    "base",
    "pandas",
    "polars",
    "sample_generators",
    "util",
    # Constants
    "TIMESERIES",
    "TIMESERIES_COLLECTIONS",
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
    # Datasets
    "damped_pendulum_ansari2023",
    "etth1",
    "etth2",
    "ettm1",
    "beijing_air_quality",
    "ettm2",
    "electricity",
    "in_silico",
    "kiwi_benchmark",
    "mimic_iv_bilos2021",
    "physionet2012",
    "physionet2019",
    "traffic",
    "ushcn",
    "ushcn_de_brouwer2019",
]

from . import base, pandas, polars, sample_generators, util
from .base import TimeSeries, TimeSeriesCollection
from .pandas import (
    TIMESERIES,
    TIMESERIES_COLLECTIONS,
    PandasTS,
    PandasTSC,
    beijing_air_quality,
    damped_pendulum_ansari2023,
    electricity,
    etth1,
    etth2,
    ettm1,
    ettm2,
    in_silico,
    kiwi_benchmark,
    mimic_iv_bilos2021,
    physionet2012,
    physionet2019,
    traffic,
    ushcn,
    ushcn_de_brouwer2019,
)
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
