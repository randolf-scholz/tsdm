r"""Utilities for time series."""

__all__ = [
    # Constants
    "TIMESERIES",
    "TIMESERIES_COLLECTIONS",
    # ABCs & Protocols
    "PandasTS",
    "PandasTSC",
    "TimeSeriesSampleGenerator",
    "FixedSliceSampleGenerator",
    # Classes
    "DampedPendulum_Ansari2023",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "Electricity",
    "InSilico",
    "KiwiBenchmark",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV_Bilos2021",
    "PhysioNet2012",
    "PhysioNet2019",
    "Traffic",
    "USHCN",
    "USHCN_DeBrouwer2019",
    # classes
    "Inputs",
    "Targets",
    "Sample",
    "PlainSample",
    "TimeSeriesSample",
    "PaddedBatch",
    # Functions
    "collate_timeseries",
]

from tsdm.timeseries._timeseries import (
    USHCN,
    DampedPendulum_Ansari2023,
    Electricity,
    ETTh1,
    ETTh2,
    ETTm1,
    ETTm2,
    InSilico,
    KiwiBenchmark,
    MIMIC_III_DeBrouwer2019,
    MIMIC_IV_Bilos2021,
    PhysioNet2012,
    PhysioNet2019,
    Traffic,
    USHCN_DeBrouwer2019,
)
from tsdm.timeseries.base import (
    FixedSliceSampleGenerator,
    Inputs,
    PaddedBatch,
    PandasTS,
    PandasTSC,
    PlainSample,
    Sample,
    Targets,
    TimeSeriesSample,
    TimeSeriesSampleGenerator,
    collate_timeseries,
)
from tsdm.types.aliases import Thunk

TIMESERIES: dict[str, Thunk[PandasTS]] = {
    "ETTh1"       : ETTh1,
    "ETTh2"       : ETTh2,
    "ETTm1"       : ETTm1,
    "ETTm2"       : ETTm2,
    "Electricity" : Electricity,
    "Traffic"     : Traffic,
}  # fmt: skip
r"""Dictionary of all available time series datasets."""

TIMESERIES_COLLECTIONS: dict[str, Thunk[PandasTSC]] = {
    "DampedPendulum_Ansari2023" : DampedPendulum_Ansari2023,
    "InSilico"                  : InSilico,
    "KiwiBenchmark"             : KiwiBenchmark,
    "MIMIC_III_DeBrouwer2019"   : MIMIC_III_DeBrouwer2019,
    "MIMIC_IV_Bilos2021"        : MIMIC_IV_Bilos2021,
    "PhysioNet2012"             : PhysioNet2012,
    "PhysioNet2019"             : PhysioNet2019,
    "USHCN"                     : USHCN,
    "USHCN_DeBrouwer2019"       : USHCN_DeBrouwer2019,
}  # fmt: skip
r"""Dictionary of all available time series collections."""

del Thunk
