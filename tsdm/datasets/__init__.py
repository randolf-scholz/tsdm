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
    # Datasets
    "BeijingAirQuality",
    "Electricity",
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "InSilicoData",
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
from tsdm.datasets.physionet2019 import Physionet2019
from tsdm.datasets.traffic import Traffic
from tsdm.datasets.ushcn import USHCN, USHCN_SmallChunkedSporadic

LOGGER = logging.getLogger(__name__)

Dataset = type[BaseDataset]
r"""Type hint for datasets."""

DATASETS: Final[dict[str, Dataset]] = {
    "BeijingAirQuality": BeijingAirQuality,
    "Electricity": Electricity,
    "ETTh1": ETTh1,
    "ETTh2": ETTh2,
    "ETTm1": ETTm1,
    "ETTm2": ETTm2,
    "InSilicoData": InSilicoData,
    "Physionet2019": Physionet2019,
    "USHCN": USHCN,
    "USHCN_SmallChunkedSporadic": USHCN_SmallChunkedSporadic,
    "Traffic": Traffic,
}
r"""Dictionary of all available datasets."""
