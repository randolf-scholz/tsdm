r"""Utility function to process folds and splits."""

__all__ = [
    # Functions
    "is_partition",
    "folds_as_frame",
    "folds_as_sparse_frame",
    "folds_from_groups",
    # Classes
]

from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from typing import Optional

import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series


def is_partition(
    partition: Iterable[Collection[Hashable]],
    /,
    *,
    union: Optional[Collection[Hashable]] = None,
) -> bool:
    r"""Check if partition is a valid partition of union."""
    sets = (set(p) for p in partition)
    part_union = set().union(*sets)

    if union is not None and part_union != set(union):
        return False
    return len(part_union) == sum(len(p) for p in partition)


def folds_from_groups(
    groups: Series, /, *, num_folds: int = 5, seed: Optional[int] = None, **splits: int
) -> Sequence[Mapping[str, Series]]:
    r"""Create folds from a Series of groups.

    Arguments:
        groups: Series of group labels.
        num_folds: Number of folds to create.
        seed: Seed for the random number generator.
        splits: Relative number of samples in each split.
            E.g. ``folds_from_groups(groups, train=7, valid=2, test=1)`` uses 7/10 of the
            samples for training, 2/10 for validation and 1/10 for testing.

    This is useful when the data needs to be grouped, e.g. due to replicate experiments.
    Simply use `pandas.groupby` and pass the result to this function.
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
    index: Optional[Index] = None,
    sparse: bool = False,
) -> DataFrame:
    r"""Create a table holding the fold information.

    Args:
        folds: a list of dictionaries, where each dictionary holds the split information,
               e.g. is a dictionary like {train: train_indices, test: test_indices}
               or   is a dictionary like {train: train_mask,    test: test_mask}
        sparse: If True, return a sparse table.
        index: The index of the time-series collection. Attempts to infer the index if not given.

    Returns:
        DataFrame with the schema
            index: metaindex of the time-series collection
            columns: fold number
            values: train/valid/test

    Example:
        +------+-------+-------+-------+-------+-------+
        | fold | 0     | 1     | 2     | 3     | 4     |
        +======+=======+=======+=======+=======+=======+
        | TS₁  | train | valid |  test | train | valid |
        +------+-------+-------+-------+-------+-------+
        | TS₂  | test  | train | valid | test  | train |
        +------+-------+-------+-------+-------+-------+
        | ...  |       |       |       |       |       |
        +------+-------+-------+-------+-------+-------+
        | TS₄  | valid | test  | train | valid |  test |
        +------+-------+-------+-------+-------+-------+
    """
    # test if the indices are given as boolean masks or as indices
    first_fold = next(iter(folds))
    first_split = next(iter(first_fold.values()))
    is_mask = first_split.dtype == bool

    match index, is_mask:
        case None, False:
            raise ValueError("Please provide `index` if `folds` are not boolean masks.")
        case None, True:
            idx: Index = (
                first_split.index
                if isinstance(first_split, Series)
                else Index(range(len(first_split)))
            )
        case _:
            idx = Index(index)

    fold_idx = Index(range(len(folds)), name="fold")
    splits = DataFrame(index=idx, columns=fold_idx, dtype="string")

    # construct the folds
    if is_mask:
        for k, fold in enumerate(folds):
            for key, split in fold.items():
                # NOTE: where cond is false is replaces with key
                splits[k].where(~split, key, inplace=True)
    else:
        for k, fold in enumerate(folds):
            for key, split in fold.items():
                # NOTE: where cond is false is replaces with key
                splits[k].where(~idx.isin(split), key, inplace=True)

    if not sparse:
        return splits

    return folds_as_sparse_frame(splits)


def folds_as_sparse_frame(df: DataFrame, /) -> DataFrame:
    r"""Create a sparse table holding the fold information."""
    # TODO: simplify this code. It should just be pd.concat(folds)
    # get categoricals
    categories = {col: s.astype("category").dtype.categories for col, s in df.items()}

    match df.columns:
        case MultiIndex() as multi_index:
            index_tuples = [
                (*col, cat)
                for col, cats in zip(multi_index, categories, strict=True)
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
                for col, cats in zip(index, categories, strict=True)
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
