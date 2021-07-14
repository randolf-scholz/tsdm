r"""Dataset Import Facilities.

tsdm.datasets
=============

Implement your own by subclassing :class:`BaseDataset`
"""

from tsdm.datasets.dataset import BaseDataset, DatasetMetaClass
from tsdm.datasets.electricity import Electricity
from tsdm.datasets.in_silico_data import InSilicoData

__all__ = ["DatasetMetaClass", "BaseDataset", "Electricity", "InSilicoData"]
