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

from tsdm.utils.torch._utils import (
    aggregate_and,
    aggregate_or,
    cumulative_and,
    cumulative_or,
)
from tsdm.utils.torch.generic import DatasetCollection, MappingDataset
from tsdm.utils.torch.timeseries import (
    IndexedArray,
    TimeSeriesBatch,
    TimeSeriesDataset,
    TimeSeriesTuple,
    TimeTensor,
)
