r"""Dataset Import Facilities.

Implement your own by subclassing `BaseDataset`

Basic Usage
-----------

>>> from tsdm.datasets import InSilicoData
>>> dataset = InSilicoData()

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
    # Types
    "Dataset",
    "DATASET_OBJECT",
    # Constants
    "DATASETS",
    # ABCs
    "BaseDataset",
    "SingleTableDataset",
    "MultiTableDataset",
    # Classes
    # Datasets
    "BeijingAirQuality",
    "ETT",
    "Electricity",
    "InSilicoData",
    "KiwiDataset",
    "KIWI_RUNS",
    "KIWI_RUNS_OLD",
    "MIMIC_III",
    "MIMIC_III_DeBrouwer2019",
    "MIMIC_IV",
    "MIMIC_IV_Bilos2021",
    "Physionet2019",
    "Physionet2012",
    "Traffic",
    "USHCN",
    "USHCN_DeBrouwer2019",
]

from tsdm.datasets import base
from tsdm.datasets.base import (
    DATASET_OBJECT,
    BaseDataset,
    Dataset,
    MultiTableDataset,
    SingleTableDataset,
)
from tsdm.datasets.beijing_air_quality import BeijingAirQuality
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.ett import ETT
from tsdm.datasets.in_silico_data import InSilicoData
from tsdm.datasets.kiwi_runs import KIWI_RUNS, KiwiDataset
from tsdm.datasets.kiwi_runs_old import KIWI_RUNS_OLD
from tsdm.datasets.mimic_iii import MIMIC_III
from tsdm.datasets.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
from tsdm.datasets.mimic_iv import MIMIC_IV
from tsdm.datasets.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
from tsdm.datasets.physionet2012 import Physionet2012
from tsdm.datasets.physionet2019 import Physionet2019
from tsdm.datasets.timeseries import TimeSeriesCollection
from tsdm.datasets.traffic import Traffic
from tsdm.datasets.ushcn import USHCN
from tsdm.datasets.ushcn_debrouwer2019 import USHCN_DeBrouwer2019

TSC: dict[str, type[TimeSeriesCollection]] = {
    "KiwiDataset": KiwiDataset,
}


DATASETS: dict[str, type[Dataset]] = {
    "BeijingAirQuality": BeijingAirQuality,
    "ETT": ETT,
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
    "KIWI_RUNS": KIWI_RUNS,
    "KIWI_RUNS_OLD": KIWI_RUNS_OLD,
    "MIMIC_III": MIMIC_III,
    "MIMIC_III_DeBrouwer2019": MIMIC_III_DeBrouwer2019,
    "MIMIC_IV": MIMIC_IV,
    "MIMIC_IV_Bilos2021": MIMIC_IV_Bilos2021,
    "Physionet2012": Physionet2012,
    "Physionet2019": Physionet2019,
    "Traffic": Traffic,
    "USHCN": USHCN,
    "USHCN_DeBrouwer2019": USHCN_DeBrouwer2019,
}
r"""Dictionary of all available dataset."""
