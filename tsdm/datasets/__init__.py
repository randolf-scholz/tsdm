r"""Dataset Import Facilities.

Implement your own by subclassing :class:`BaseDataset`
"""

import logging
from typing import Final, Type

from tsdm.datasets.dataset import BaseDataset, DatasetMetaClass
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.in_silico_data import InSilicoData

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = [
    "DATASETS",
    "DatasetMetaClass",
    "BaseDataset",
    "Electricity",
    "InSilicoData",
]

DATASETS: Final[dict[str, Type[BaseDataset]]] = {
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
}
