r"""Subclasses of `torch.util.data.Dataset`."""

__all__ = [
    # Modules
    "timeseries",
    # Protocols
    "MapStyleDataset",
    # Classes
    "DatasetCollection",
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
    # Functions
    # data - arrow
    "cast_columns",
    "compute_entropy",
    "filter_nulls",
    "force_cast",
    "table_info",
    # data
    "aggregate_nondestructive",
    "detect_outliers",
    "detect_outliers_dataframe",
    "detect_outliers_series",
    "float_is_int",
    "get_integer_cols",
    "joint_keys",
    "make_dataframe",
    "remove_outliers",
    "remove_outliers_dataframe",
    "remove_outliers_series",
    "strip_whitespace",
    "vlookup_uniques",
]

from tsdm.utils.data import timeseries
from tsdm.utils.data._arrow import (
    cast_columns,
    compute_entropy,
    filter_nulls,
    force_cast,
    table_info,
)
from tsdm.utils.data._data import (
    InlineTable,
    aggregate_nondestructive,
    detect_outliers,
    detect_outliers_dataframe,
    detect_outliers_series,
    float_is_int,
    get_integer_cols,
    joint_keys,
    make_dataframe,
    remove_outliers,
    remove_outliers_dataframe,
    remove_outliers_series,
    strip_whitespace,
    vlookup_uniques,
)
from tsdm.utils.data.dataloaders import (
    collate_packed,
    collate_padded,
    unpack_sequence,
    unpad_sequence,
)
from tsdm.utils.data.datasets import (
    DataFrame2Dataset,
    DatasetCollection,
    MappingDataset,
    MapStyleDataset,
)
from tsdm.utils.data.folds import (
    folds_as_frame,
    folds_as_sparse_frame,
    folds_from_groups,
    is_partition,
)
