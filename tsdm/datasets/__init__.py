r"""Dataset Import Facilities.

Implement your own by subclassing :class:`BaseDataset`
"""

from __future__ import annotations

__all__ = [
    # Meta-Objects
    "Dataset",
    "DatasetType",
    "DATASETS",
    # Classes
    "DatasetMetaClass",
    "BaseDataset",
    "SequenceDataset",
    # Datasets
    "BeijingAirQuality",
    "Electricity",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "InSilicoData",
    "MIMIC_III",
    "Physionet2019",
    "USHCN",
    "USHCN_SmallChunkedSporadic",
    "Traffic",
]


import logging
from typing import Final

from tsdm.datasets.beijing_air_quality import BeijingAirQuality
from tsdm.datasets.dataset import BaseDataset, DatasetMetaClass, SequenceDataset
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.etdataset import ETTh1, ETTh2, ETTm1, ETTm2
from tsdm.datasets.in_silico_data import InSilicoData
from tsdm.datasets.mimic_iii import MIMIC_III
from tsdm.datasets.physionet2019 import Physionet2019
from tsdm.datasets.traffic import Traffic
from tsdm.datasets.ushcn import USHCN, USHCN_SmallChunkedSporadic

from tsdm.util.types import ModularLookupTable

LOGGER = logging.getLogger(__name__)

Dataset = BaseDataset
r"""Type hint for datasets."""

DatasetType = type[BaseDataset]
r"""Type hint for datasets."""

DATASETS: Final[ModularLookupTable[DatasetType]] = {
    "BeijingAirQuality": BeijingAirQuality,
    "Electricity": Electricity,
    "ETTh1": ETTh1,
    "ETTh2": ETTh2,
    "ETTm1": ETTm1,
    "ETTm2": ETTm2,
    "InSilicoData": InSilicoData,
    "MIMIC_III": MIMIC_III,
    "Physionet2019": Physionet2019,
    "USHCN": USHCN,
    "USHCN_SmallChunkedSporadic": USHCN_SmallChunkedSporadic,
    "Traffic": Traffic,
}
r"""Dictionary of all available datasets."""
