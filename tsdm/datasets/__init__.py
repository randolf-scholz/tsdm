"""
tsdm.datasets
=============

Implement your own by subclassing :class:`BaseDataset`
"""

from .dataset import DatasetMetaClass, BaseDataset
from .electricity import Electricity
from .in_silico_data import InSilicoData

__all__ = ['DatasetMetaClass', 'BaseDataset', 'Electricity', 'InSilicoData']
