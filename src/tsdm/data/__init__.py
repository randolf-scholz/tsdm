r"""Subclasses of `torch.util.data.Dataset`."""

__all__ = [
    # Modules
    # "timeseries",
    # Protocols
    "TorchDataset",
    "MapDataset",
    "IterableDataset",
    "PandasDataset",
    "IndexableDataset",
    "Dataset",
    # Classes
    "MappingDataset",
    "InlineTable",
    "DataFrame2Dataset",
    # folds
    "is_partition",
    "folds_as_frame",
    "folds_as_sparse_frame",
    "folds_from_groups",
    # rnn
    "collate_packed",
    "collate_padded",
    "unpad_sequence",
    "unpack_sequence",
    # timeseries
    "TimeSeriesCollection",
    "TimeSeriesDataset",
    "TimeSeriesSampleGenerator",
    # Functions
    # data - arrow
    # data
    "aggregate_nondestructive",
    "detect_outliers",
    "float_is_int",
    "get_integer_cols",
    "make_dataframe",
    "remove_outliers",
    "strip_whitespace",
]

from tsdm.data._data import (
    InlineTable,
    aggregate_nondestructive,
    detect_outliers,
    float_is_int,
    get_integer_cols,
    make_dataframe,
    remove_outliers,
    strip_whitespace,
)
from tsdm.data.dataloaders import (
    collate_packed,
    collate_padded,
    unpack_sequence,
    unpad_sequence,
)
from tsdm.data.datasets import (
    DataFrame2Dataset,
    Dataset,
    IndexableDataset,
    IterableDataset,
    MapDataset,
    MappingDataset,
    PandasDataset,
    TorchDataset,
)
from tsdm.data.folds import (
    folds_as_frame,
    folds_as_sparse_frame,
    folds_from_groups,
    is_partition,
)
from tsdm.data.timeseries import (
    TimeSeriesCollection,
    TimeSeriesDataset,
    TimeSeriesSampleGenerator,
)
