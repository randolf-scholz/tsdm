r"""Dataset Import Facilities.

Implement your own by subclassing :class:`BaseDataset`
"""

from __future__ import annotations

__all__ = [
    # Meta-Objects
    "Dataset",
    "DATASETS",
    # Classes
    "DatasetMetaClass",
    "BaseDataset",
    "SequenceDataset",
    "DataSetCollection",
    # Datasets
    "BeijingAirQuality",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
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
from typing import Final, Union

from tsdm.datasets.beijing_air_quality import BeijingAirQuality
from tsdm.datasets.dataset import (
    BaseDataset,
    DataSetCollection,
    DatasetMetaClass,
    SequenceDataset,
)
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.etdataset import ETTh1, ETTh2, ETTm1, ETTm2
from tsdm.datasets.in_silico_data import InSilicoData
from tsdm.datasets.kiwi_runs import KIWI_RUNS
from tsdm.datasets.mimic_iii import MIMIC_III
from tsdm.datasets.physionet2019 import Physionet2019
from tsdm.datasets.traffic import Traffic
from tsdm.datasets.ushcn import USHCN, USHCN_SmallChunkedSporadic
from tsdm.util.types import LookupTable

__logger__ = logging.getLogger(__name__)

Dataset = Union[BaseDataset, type[BaseDataset]]
r"""Type hint for datasets."""

DATASETS: Final[LookupTable[Dataset]] = {
    "BeijingAirQuality": BeijingAirQuality,
    "ETTh1": ETTh1,
    "ETTh2": ETTh2,
    "ETTm1": ETTm1,
    "ETTm2": ETTm2,
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
    "KIWI_RUNS_TASK": KIWI_RUNS,
    "MIMIC_III": MIMIC_III,
    "Physionet2019": Physionet2019,
    "Traffic": Traffic,
    "USHCN": USHCN,
    "USHCN_SmallChunkedSporadic": USHCN_SmallChunkedSporadic,
}
r"""Dictionary of all available datasets."""
