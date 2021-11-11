r"""Base Classes for datasets."""


__all__ = [
    # Classes
    "BaseDataset",
    "DatasetCollection",
    "DatasetMetaClass",
    "SequenceDataset",
    "TimeTensor",
    "TimeSeriesDataset",
    "Template",
    # Types
    "IndexedArray",
    # Functions
    "tensor_info",
]


import logging

from tsdm.datasets.base.dataset import (
    BaseDataset,
    DatasetCollection,
    DatasetMetaClass,
    SequenceDataset,
)
from tsdm.datasets.base.template import Template
from tsdm.datasets.base.timeseries import (
    IndexedArray,
    TimeSeriesDataset,
    TimeTensor,
    tensor_info,
)

__logger__ = logging.getLogger(__name__)
