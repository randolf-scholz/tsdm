r"""Utility function to process folds and splits."""

__all__ = [
    # functions
    "folds_as_frame",
    "fold_frame_sparse",
]

from collections.abc import Mapping, Sequence
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame, Index, MultiIndex, Series

FOLDS = Sequence[Mapping[Literal["train", "valid", "test"], Union[Series, NDArray]]]


def folds_as_frame(
    folds: FOLDS, *, index: Optional[Mapping | NDArray] = None, sparse: bool = False
) -> DataFrame:
    r"""Create a table holding the fold information..

    +-------+-------+-------+-------+-------+-------+
    | fold  | 0     | 1     | 2     | 3     | 4     |
    +=======+=======+=======+=======+=======+=======+
    | 15325 | train | train |  test | train | train |
    +-------+-------+-------+-------+-------+-------+
    | 15326 | train | train | train | train |  test |
    +-------+-------+-------+-------+-------+-------+
    | 15327 |  test | valid | train | train | train |
    +-------+-------+-------+-------+-------+-------+
    | 15328 | valid | train | train | train |  test |
    +-------+-------+-------+-------+-------+-------+
    """
    if index is None:
        # create a default index
        first_fold = next(iter(folds))
        first_split = next(iter(first_fold.values()))
        index = (
            first_split.index
            if isinstance(first_split, Series)
            else np.arange(len(first_split))
        )

    fold_idx = Index(list(range(len(folds))), name="fold")
    splits = DataFrame(index=index, columns=fold_idx, dtype="string")

    for k in fold_idx:
        for key, split in folds[k].items():
            # where cond is false is replaces with key
            splits[k] = splits[k].where(~split, key)

    if not sparse:
        return splits
    return fold_frame_sparse(splits)


def fold_frame_sparse(df: DataFrame, /) -> DataFrame:
    r"""Create a sparse table holding the fold information."""
    columns = df.columns

    # get categoricals
    categories = {col: df[col].astype("category").dtype.categories for col in columns}

    if isinstance(df.columns, MultiIndex):
        index_tuples = [
            (*col, cat)
            for col, cats in zip(columns, categories)
            for cat in categories[col]
        ]
        names = df.columns.names + ["partition"]
    else:
        index_tuples = [
            (col, cat)
            for col, cats in zip(columns, categories)
            for cat in categories[col]
        ]
        names = [df.columns.name, "partition"]

    new_columns = MultiIndex.from_tuples(index_tuples, names=names)
    result = DataFrame(index=df.index, columns=new_columns, dtype=bool)

    if isinstance(df.columns, MultiIndex):
        for col in new_columns:
            result[col] = df[col[:-1]] == col[-1]
    else:
        for col in new_columns:
            result[col] = df[col[0]] == col[-1]

    return result
