r"""Base Classes for dataset."""

__all__ = [
    # Types
    # ABCs
    "BaseDataset",
    "Dataset",
    # Classes
    "SimpleDataset",
    "Template",
]

from tsdm.datasets.base._base import BaseDataset, Dataset, SimpleDataset
from tsdm.datasets.base.template import Template
