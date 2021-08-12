r"""Dataset Import Facilities.

Implement your own by subclassing :class:`BaseDataset`
"""

from tsdm.datasets.dataset import BaseDataset, DatasetMetaClass
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.in_silico_data import InSilicoData

__all__ = [
    "DATASETS",
    "DatasetMetaClass",
    "BaseDataset",
    "Electricity",
    "InSilicoData",
]

DATASETS = {
    "Electricity": Electricity,
    "InSilicoData": InSilicoData,
}
