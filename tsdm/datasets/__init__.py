r"""Dataset Import Facilities.

Implement your own by subclassing :class:`BaseDataset`
"""

import logging
from typing import Final, Type

from tsdm.datasets.beijing_air_quality import BeijingAirQuality
from tsdm.datasets.dataset import BaseDataset, DatasetMetaClass
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.in_silico_data import InSilicoData
from tsdm.datasets.physionet2019 import Physionet2019

LOGGER = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "Dataset",
    "DATASETS",
    "DatasetMetaClass",
    "BaseDataset",
    "Electricity",
    "InSilicoData",
    "Physionet2019",
    "BeijingAirQuality",
]

Dataset = Type[BaseDataset]
r"""Type hint for datasets."""

DATASETS: Final[dict[str, Dataset]] = {
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
    "Physionet2019": Physionet2019,
    "BeijingAirQuality": BeijingAirQuality,
}
r"""Dictionary containing all available datasets."""
