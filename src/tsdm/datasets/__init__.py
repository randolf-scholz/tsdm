r"""Dataset Import Facilities.

This package contains a collection of datasets for time series analysis.
The classes defined here are focused on dataset loading, pre-processing, and serialization.
For actual usage of the datasets in machine learning models, please use the wrapped
classes in the `tsdm.timeseries` package.


Basic Usage
-----------

>>> from tsdm.datasets import InSilico
>>> dataset = InSilico()

Some design decisions:

1. Why not use `Series` instead of Mapping for dataset?
    - `Series[object]` has bad performance issues in construction.
2. Should we have Dataset style iteration or dict style iteration?
    - Note that for `dict`, `iter(dict)` iterates over index.
    - For `Series`, `DataFrame`, `TorchDataset`, `__iter__` iterates over values.
"""

__all__ = [
    # Sub-Modules
    "base",
    "uci",
    "mimic",
    "physionet",
    "kiwi",
    "synthetic",
    "ushcn",
    # Constants
    "DATASETS",
    # ABCs & Protocols
    "Dataset",
    "DatasetBase",
    # Classes
    "DampedPendulum_Ansari2023",
    "BeijingAirQuality",
    "ETT",
    "Electricity",
    "InSilico",
    "KiwiBenchmark",
    "MIMIC_III",
    "MIMIC_III_RAW",
    "MIMIC_III_Bilos2021",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV",
    "MIMIC_IV_RAW",
    "MIMIC_IV_Bilos2021",
    "PhysioNet2019",
    "PhysioNet2012",
    "Traffic",
    "USHCN",
    "USHCN_DeBrouwer2019",
]

from tsdm.datasets import (
    base,
    kiwi,
    mimic,
    physionet,
    synthetic,
    uci,
    ushcn,
)
from tsdm.datasets.base import (
    Dataset,
    DatasetBase,
)
from tsdm.datasets.ett import ETT
from tsdm.datasets.kiwi import (
    InSilico,
    KiwiBenchmark,
)
from tsdm.datasets.mimic import (
    MIMIC_III,
    MIMIC_III_RAW,
    MIMIC_IV,
    MIMIC_IV_RAW,
    MIMIC_III_Bilos2021,
    MIMIC_III_DeBrouwer2019,
    MIMIC_IV_Bilos2021,
)
from tsdm.datasets.physionet import PhysioNet2012, PhysioNet2019
from tsdm.datasets.synthetic import DampedPendulum_Ansari2023
from tsdm.datasets.uci import BeijingAirQuality, Electricity, Traffic
from tsdm.datasets.ushcn import USHCN, USHCN_DeBrouwer2019

DATASETS: dict[str, type[Dataset]] = {
    "BeijingAirQuality"         : BeijingAirQuality,
    "DampedPendulum_Ansari2023" : DampedPendulum_Ansari2023,
    "ETT"                       : ETT,
    "Electricity"               : Electricity,
    "InSilico"                  : InSilico,
    "KiwiBenchmark"             : KiwiBenchmark,
    "MIMIC_III"                 : MIMIC_III,
    "MIMIC_III_Bilos2021"       : MIMIC_III_Bilos2021,
    "MIMIC_III_DeBrouwer2019"   : MIMIC_III_DeBrouwer2019,
    "MIMIC_III_RAW"             : MIMIC_III_RAW,
    "MIMIC_IV"                  : MIMIC_IV,
    "MIMIC_IV_Bilos2021"        : MIMIC_IV_Bilos2021,
    "MIMIC_IV_RAW"              : MIMIC_IV_RAW,
    "PhysioNet2012"             : PhysioNet2012,
    "PhysioNet2019"             : PhysioNet2019,
    "Traffic"                   : Traffic,
    "USHCN"                     : USHCN,
    "USHCN_DeBrouwer2019"       : USHCN_DeBrouwer2019,
}  # fmt: skip
r"""Dictionary of all available dataset."""
