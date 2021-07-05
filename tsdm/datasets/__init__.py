r"""Dataset Import Facilities.

tsdm.datasets
=============

Implement your own by subclassing :class:`BaseDataset`
"""

from .dataset import BaseDataset, DatasetMetaClass
from .electricity import Electricity
from .in_silico_data import InSilicoData

__all__ = ["DatasetMetaClass", "BaseDataset", "Electricity", "InSilicoData"]
