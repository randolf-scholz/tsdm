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
    "damped_pendulum_ansari2023",
    "etth1",
    "etth2",
    "ettm1",
    "ettm2",
    "electricity",
    "in_silico",
    "kiwi_benchmark",
    "mimic_iii_de_brouwer2019",
    "mimic_iv_bilos2021",
    "physio_net2012",
    "physio_net2019",
    "traffic",
    "ushcn",
    "ushcn_de_brouwer2019",
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
    physio_net2012,
    physio_net2019,
    traffic,
    ushcn,
    ushcn_de_brouwer2019,
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
    "ETTh1"       : etth1,
    "ETTh2"       : etth2,
    "ETTm1"       : ettm1,
    "ETTm2"       : ettm2,
    "Electricity" : electricity,
    "Traffic"     : traffic,
}  # fmt: skip
r"""Dictionary of all available time series datasets."""

TIMESERIES_COLLECTIONS: dict[str, Thunk[PandasTSC]] = {
    "DampedPendulum_Ansari2023" : damped_pendulum_ansari2023,
    "InSilico"                  : in_silico,
    "KiwiBenchmark"             : kiwi_benchmark,
    "MIMIC_III_DeBrouwer2019"   : mimic_iii_de_brouwer2019,
    "MIMIC_IV_Bilos2021"        : mimic_iv_bilos2021,
    "PhysioNet2012"             : physio_net2012,
    "PhysioNet2019"             : physio_net2019,
    "USHCN"                     : ushcn,
    "USHCN_DeBrouwer2019"       : ushcn_de_brouwer2019,
}  # fmt: skip
r"""Dictionary of all available time series collections."""

del Thunk
