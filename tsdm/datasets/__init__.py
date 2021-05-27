"""
tsdm.datasets
=============

Implement your own by subclassing :class:`BaseDataset`
"""

from .dataset import DatasetMetaClass, BaseDataset
from .electricity import Electricity

__all__ = ['DatasetMetaClass', 'BaseDataset', 'Electricity']
