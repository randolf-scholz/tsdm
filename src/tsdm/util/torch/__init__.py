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
    "aggregate_and",
    "aggregate_or",
    "cumulative_and",
    "cumulative_or",
]

from tsdm.util.torch._utils import (
    aggregate_and,
    aggregate_or,
    cumulative_and,
    cumulative_or,
)
from tsdm.util.torch.generic import DatasetCollection, MappingDataset
from tsdm.util.torch.timeseries import (
    IndexedArray,
    TimeSeriesBatch,
    TimeSeriesDataset,
    TimeSeriesTuple,
    TimeTensor,
)
