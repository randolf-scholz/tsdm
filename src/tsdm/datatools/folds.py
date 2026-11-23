r"""Utility functions for processing folds and splits."""

__all__ = [
    # Functions
    "folds_as_frame",
    "folds_as_sparse_frame",
    "folds_from_groups",
]

from collections.abc import Mapping, Sequence
from typing import SupportsIndex

import numpy as np
import pandas as pd
from pandas import DataFrame, Index, MultiIndex, Series

from .collections import IndexableDataset


def folds_from_groups(
    groups: Series, /, *, num_folds: int = 5, seed: int | None = None, **splits: int
) -> Sequence[Mapping[str, Series]]:
    r"""Create folds while keeping samples with the same group label together.

    Pass a Series with one group label per sample, for example a replicate
    identifier for each measurement. Split weights apply to groups; sample
    counts can differ from these proportions when group sizes differ.

    Args:
        groups: Series of group labels.
        num_folds: Number of folds to create.
        seed: Seed for the random number generator.
        **splits: Relative number of group chunks in each split.
            For example, `folds_from_groups(groups, train=7, valid=2, test=1)`
            assigns seven of ten chunks to training, two to validation, and one
            to testing. Chunks contain approximately equal numbers of groups.

    Returns:
        One mapping per fold, from split names to boolean Series aligned with
        `groups`. All samples with the same group label belong to the same split.

    Raises:
        ValueError: If no splits are provided or the sum of split weights is not
            a multiple of `num_folds`.
    """
    if not splits:
        raise ValueError("No splits provided!")

    num_chunks = sum(splits.values())
    q, remainder = divmod(num_chunks, num_folds)

    if remainder:
        raise ValueError("Sum of chunks must be a multiple of num_folds!")

    unique_groups = groups.unique()
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(unique_groups)
    chunks = np.array(np.array_split(shuffled, num_chunks), dtype=object)

    slices, a, b = {}, 0, 0
    for key, size in splits.items():
        a, b = b, b + size
        slices[key] = np.arange(a, b)

    folds = []
    for k in range(num_folds):
        fold = {}
        for key in splits:
            mask = (slices[key] + q * k) % num_chunks
            selection = chunks[mask]
            chunk = np.concatenate(selection)
            fold[key] = groups.isin(chunk)
        folds.append(fold)

    return folds


def folds_as_frame(
    folds: Sequence[Mapping[str, Series]],
    /,
    *,
    index: IndexableDataset[SupportsIndex] | None = None,
    sparse: bool = False,
) -> DataFrame:
    r"""Create a table holding the fold information.

    Args:
        folds: Non-empty sequence of non-empty mappings from split names to
            sample indices or boolean masks. All splits must use the same
            representation. For example, `{"train": train_mask, "test": test_mask}`.
        index: Sample index. If omitted, infer it from the first fold's indices
            or the first mask's index, using positional indices for array masks.
        sparse: If `True`, return boolean split-membership columns instead of
            split labels.

    Returns:
        A DataFrame indexed by sample, with one column per fold and split names
        as values. Unassigned samples have missing values; later splits overwrite
        earlier ones if they overlap. With `sparse=True`, columns have a two-level
        index of fold number and split name, and values indicate membership.

    Examples:
        A table of split labels with one column per fold:

        | Sample | 0     | 1     | 2     | 3     | 4     |
        | ------ | ----- | ----- | ----- | ----- | ----- |
        | TS₁    | train | valid | test  | train | valid |
        | TS₂    | test  | train | valid | test  | train |
        | …      |       |       |       |       |       |
        | TS₄    | valid | test  | train | valid | test  |
    """
    # test if the indices are given as boolean masks or as indices
    first_fold = next(iter(folds))
    first_split = next(iter(first_fold.values()))
    is_mask = pd.api.types.is_bool_dtype(np.array(first_split))

    if index is None and not is_mask:
        # infer the index from the union of all indices in the folds
        index = np.unique([*split for split in first_fold.values()])  # pyrefly: ignore[no-matching-overload]

    idx = (
        Index(index)
        if index is not None
        else (
            first_split.index
            if isinstance(first_split, Series)
            else Index(range(len(first_split)))
        )
    )

    fold_idx = Index(range(len(folds)), name="fold")
    splits = DataFrame(index=idx, columns=fold_idx, dtype="string")

    # construct the folds
    if is_mask:
        for k, fold in enumerate(folds):
            for key, split in fold.items():
                # NOTE: where cond is false is replaces with key
                splits[k] = splits[k].where(~np.asarray(split), key)
    else:
        for k, fold in enumerate(folds):
            for key, split in fold.items():
                # NOTE: where cond is false is replaces with key
                splits[k] = splits[k].where(~idx.isin(split), key)

    if not sparse:
        return splits

    return folds_as_sparse_frame(splits)


def folds_as_sparse_frame(df: DataFrame, /) -> DataFrame:
    r"""Expand fold labels into boolean split-membership columns.

    Append a `partition` level to the column index, with one column for each
    observed split label in each fold. Missing labels do not create a partition.
    The result uses boolean columns rather than pandas sparse dtypes.
    """
    # TODO: simplify this code. It should just be pd.concat(folds)
    # get categoricals
    categories = {col: s.astype("category").cat.categories for col, s in df.items()}

    match df.columns:
        case MultiIndex() as multi_index:
            index_tuples = [
                (*col, cat)
                for col, _ in zip(multi_index, categories, strict=True)
                for cat in categories[col]
            ]
            names = [*df.columns.names, "partition"]
            new_columns = MultiIndex.from_tuples(index_tuples, names=names)
            result = DataFrame(index=df.index, columns=new_columns, dtype=bool)

            for col in new_columns:
                result[col] = df[col[:-1]] == col[-1]
            return result

        case Index() as index:
            index_tuples = [
                (col, cat)
                for col, _ in zip(index, categories, strict=True)
                for cat in categories[col]
            ]
            names = [df.columns.name, "partition"]
            new_columns = MultiIndex.from_tuples(index_tuples, names=names)
            result = DataFrame(index=df.index, columns=new_columns, dtype=bool)

            for col in new_columns:
                result[col] = df[col[0]] == col[-1]
            return result

        case _:
            raise TypeError(f"Column type {type(df.columns)} not supported.")
