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

import logging

from tsdm.datasets.base._base import BaseDataset, Dataset, SimpleDataset
from tsdm.datasets.base.template import Template

__logger__ = logging.getLogger(__name__)
