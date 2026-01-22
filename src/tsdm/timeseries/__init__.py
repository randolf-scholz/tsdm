r"""Utilities for time series."""

__all__ = [
    # submodules:
    "base",
    "pandas",
    "sampling",
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
    "mimic_iii_de_brouwer2019",
    "mimic_iv_bilos2021",
    "physionet2012",
    "physionet2019",
    "traffic",
    "ushcn",
    "ushcn_de_brouwer2019",
]

from collections.abc import Callable as Fn
from typing import Any

from pandas import DataFrame

from tsdm.timeseries import base, pandas, sampling, util
from tsdm.timeseries.base import TimeSeries, TimeSeriesCollection
from tsdm.timeseries.pandas import (
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
    mimic_iii_de_brouwer2019,
    mimic_iv_bilos2021,
    physionet2012,
    physionet2019,
    traffic,
    ushcn,
    ushcn_de_brouwer2019,
)
from tsdm.timeseries.sampling import (
    FixedSliceSampleGenerator,
    Inputs,
    PlainSample,
    Sample,
    Targets,
    TimeSeriesSampleGenerator,
)
from tsdm.timeseries.util import PaddedBatch, TimeSeriesSample, collate_timeseries

TIMESERIES: dict[str, Fn[[], TimeSeries[DataFrame]]] = {
    "ETTh1"       : etth1,
    "ETTh2"       : etth2,
    "ETTm1"       : ettm1,
    "ETTm2"       : ettm2,
    "Electricity" : electricity,
    "Traffic"     : traffic,
}  # fmt: skip
r"""Dictionary of all available time series datasets."""

TIMESERIES_COLLECTIONS: dict[str, Fn[[], TimeSeriesCollection[Any, DataFrame]]] = {
    "DampedPendulum_Ansari2023" : damped_pendulum_ansari2023,
    "InSilico"                  : in_silico,
    "KiwiBenchmark"             : kiwi_benchmark,
    "MIMIC_III_DeBrouwer2019"   : mimic_iii_de_brouwer2019,
    "MIMIC_IV_Bilos2021"        : mimic_iv_bilos2021,
    "PhysioNet2012"             : physionet2012,
    "PhysioNet2019"             : physionet2019,
    "USHCN"                     : ushcn,
    "BeijingAirQuality"         : beijing_air_quality,
    "USHCN_DeBrouwer2019"       : ushcn_de_brouwer2019,
}  # fmt: skip
r"""Dictionary of all available time series collections."""
