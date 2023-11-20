r"""Dataset Import Facilities.

Implement your own by subclassing `BaseDataset`

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
    "timeseries",
    "uci",
    "mimic",
    "physionet",
    "kiwi",
    "ushcn",
    # Types
    "Dataset",
    # Constants
    "DATASETS",
    # ABCs
    "BaseDataset",
    "SingleTableDataset",
    "MultiTableDataset",
    # Classes
    # Datasets
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
    # Timeseries
    "InSilicoTSC",
    "KiwiBenchmarkTSC",
    # Deprecated Datasets
    "KiwiRuns",
    # Deprecated Timeseries
    "KiwiRunsTSC",
]

# submodules
from tsdm.datasets import base, kiwi, mimic, physionet, timeseries, uci, ushcn
from tsdm.datasets.base import (
    BaseDataset,
    Dataset,
    MultiTableDataset,
    SingleTableDataset,
)
from tsdm.datasets.ett import ETT
from tsdm.datasets.kiwi import (
    InSilico,
    InSilicoTSC,
    KiwiBenchmark,
    KiwiBenchmarkTSC,
    KiwiRuns,
    KiwiRunsTSC,
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
    "BeijingAirQuality": BeijingAirQuality,
    "DampedPendulum_Ansari2023": DampedPendulum_Ansari2023,
    "ETT": ETT,
    "Electricity": Electricity,
    "InSilico": InSilico,
    "KiwiBenchmark": KiwiBenchmark,
    "MIMIC_III": MIMIC_III_RAW,
    "MIMIC_III_DeBrouwer2019": MIMIC_III_DeBrouwer2019,
    "MIMIC_III_Bilos2021": MIMIC_III_Bilos2021,
    "MIMIC_IV": MIMIC_IV_RAW,
    "MIMIC_IV_Bilos2021": MIMIC_IV_Bilos2021,
    "Physionet2012": PhysioNet2012,
    "Physionet2019": PhysioNet2019,
    "Traffic": Traffic,
    "USHCN": USHCN,
    "USHCN_DeBrouwer2019": USHCN_DeBrouwer2019,
}
r"""Dictionary of all available dataset."""
