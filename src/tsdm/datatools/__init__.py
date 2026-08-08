r"""Subclasses of `torch.util.data.Dataset`."""

__all__ = [
    # Modules
    # "timeseries",
    # Protocols
    "TorchDataset",
    "MapDataset",
    "IterableDataset",
    "PandasDataset",
    "Indexable",
    "Dataset",
    # Classes
    "MappingDataset",
    "InlineTable",
    # folds
    "is_partition",
    "folds_as_frame",
    "folds_as_sparse_frame",
    "folds_from_groups",
    # Functions
    # data - arrow
    # data
    "aggregate_nondestructive",
    "select_outliers",
    "is_integer_series",
    "get_integer_cols",
    "make_dataframe",
    "remove_outliers",
    "strip_whitespace",
]

from .collections import (
    Dataset,
    Indexable,
    IterableDataset,
    MapDataset,
    MappingDataset,
    PandasDataset,
    TorchDataset,
)
from .folds import (
    folds_as_frame,
    folds_as_sparse_frame,
    folds_from_groups,
    is_partition,
)
from .preprocessing import (
    aggregate_nondestructive,
    get_integer_cols,
    is_integer_series,
    remove_outliers,
    select_outliers,
    strip_whitespace,
)
from .serialize import InlineTable, make_dataframe
