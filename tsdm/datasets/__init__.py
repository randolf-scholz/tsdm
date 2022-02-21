r"""Dataset Import Facilities.

Implement your own by subclassing :class:`BaseDataset`
"""

__all__ = [
    # Sub-Packages
    "base",
    "synthetic",
    "torch",
    # Types
    "DATASET",
    # Constants
    "DATASETS",
    # ABCs
    "BaseDataset",
    "Dataset",
    "Template",
    # Classes
    "IndexedArray",
    "TimeSeriesDataset",
    "TimeTensor",
    # Datasets
    "BeijingAirQuality",
    "ETT",
    "Electricity",
    "InSilicoData",
    "KIWI_RUNS",
    "MIMIC_III",
    "Physionet2019",
    "Traffic",
    "USHCN",
    "USHCN_SmallChunkedSporadic",
]


import logging
from typing import Final

from tsdm.datasets import base, synthetic, torch
from tsdm.datasets.base import BaseDataset, Dataset, Template
from tsdm.datasets.beijing_air_quality import BeijingAirQuality
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.ett import ETT
from tsdm.datasets.in_silico_data import InSilicoData
from tsdm.datasets.kiwi_runs import KIWI_RUNS
from tsdm.datasets.mimic_iii import MIMIC_III
from tsdm.datasets.physionet2019 import Physionet2019
from tsdm.datasets.torch import IndexedArray, TimeSeriesDataset, TimeTensor
from tsdm.datasets.traffic import Traffic
from tsdm.datasets.ushcn import USHCN, USHCN_SmallChunkedSporadic
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

DATASET = BaseDataset
r"""Type hint for dataset."""

DATASETS: Final[LookupTable[type[DATASET]]] = {
    "BeijingAirQuality": BeijingAirQuality,
    "ETT": ETT,
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
    "KIWI_RUNS_TASK": KIWI_RUNS,
    "MIMIC_III": MIMIC_III,
    "Physionet2019": Physionet2019,
    "Traffic": Traffic,
    "USHCN": USHCN,
    "USHCN_SmallChunkedSporadic": USHCN_SmallChunkedSporadic,
}
r"""Dictionary of all available dataset."""
