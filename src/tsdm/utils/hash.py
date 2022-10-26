r"""Hash function utils."""

__all__ = [
    # Functions
    "hash_set",
    "hash_pandas",
]

from collections import Counter
from collections.abc import Hashable, Iterable

import pandas
from pandas import DataFrame, Index, Series


def hash_set(x: Iterable[Hashable], /, *, ignore_duplicates: bool = False) -> int:
    r"""Compute a permutation invariant hash of an iterable."""
    if ignore_duplicates:
        return hash(frozenset(x))
    # We use counter as a proxy for multisets, do deal with duplicates in x.
    mdict = Counter((hash(y) for y in x))
    return hash(frozenset(mdict.items()))


def hash_pandas(
    df: Index | Series | DataFrame,
    /,
    *,
    index: bool = True,
    ignore_duplicates: bool = False,
    row_invariant: bool = False,
    col_invariant: bool = False,
) -> int:
    r"""Hash pandas object to a single number.

    If row_invariant is True, then the hash is invariant to the order of the rows.
    If col_invariant is True, then the hash is invariant to the order of the columns.

    .. math:: hash(PX) = hash(X) and hash(XQ) = hash(X)

    for permutation matrices $P$ and $Q$ respectively.
    """
    # TODO: problem: duplicate rows ⇝ include standard index for rows/cols!
    if col_invariant:
        raise NotImplementedError("col_invariant is not implemented yet.")
    row_hash = pandas.util.hash_pandas_object(df, index=index)
    if row_invariant:
        return hash_set(row_hash, ignore_duplicates=ignore_duplicates)
    return hash(tuple(row_hash))
