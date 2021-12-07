r"""Base Classes for dataset."""

__all__ = [
    # Types
    "IndexedArray",
    # ABCs
    "BaseDataset",
    "Dataset",
    # Classes
    "DatasetCollection",
    "SequenceDataset",
    "SimpleDataset",
    "TimeTensor",
    "TimeSeriesDataset",
    "TimeSeriesTuple",
    "TimeSeriesBatch",
    "Template",
    # Functions
    "tensor_info",
]

import logging

from tsdm.datasets.base._base import BaseDataset, Dataset, SimpleDataset
from tsdm.datasets.base.template import Template
from tsdm.datasets.base.timeseries import (
    IndexedArray,
    TimeSeriesBatch,
    TimeSeriesDataset,
    TimeSeriesTuple,
    TimeTensor,
    tensor_info,
)
from tsdm.datasets.base.torch_datasets import DatasetCollection, SequenceDataset

__logger__ = logging.getLogger(__name__)
