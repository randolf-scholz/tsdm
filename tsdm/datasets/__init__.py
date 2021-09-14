r"""Dataset Import Facilities.

Implement your own by subclassing :class:`BaseDataset`
"""

import logging
from typing import Final, Type

from tsdm.datasets.dataset import BaseDataset, DatasetMetaClass
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.in_silico_data import InSilicoData
from tsdm.datasets.physionet2019 import Physionet2019

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "DATASETS",
    "DatasetMetaClass",
    "BaseDataset",
    "Electricity",
    "InSilicoData",
    "Physionet2019",
]

DATASETS: Final[dict[str, Type[BaseDataset]]] = {
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
    "Physionet2019": Physionet2019,
}
