r"""Subclasses of `torch.util.data.Dataset`."""

__all__ = [
    # Types
    "IndexedArray",
    # Classes
    "TimeTensor",
    "TimeSeriesDataset",
    "TimeSeriesTuple",
    "TimeSeriesBatch",
    "DatasetCollection",
    "MappingDataset",
    # Functions
]

import logging

from tsdm.datasets.torch.generic import DatasetCollection, MappingDataset
from tsdm.datasets.torch.timeseries import (
    IndexedArray,
    TimeSeriesBatch,
    TimeSeriesDataset,
    TimeSeriesTuple,
    TimeTensor,
)

__logger__ = logging.getLogger(__name__)
