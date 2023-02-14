r"""Hash function utils."""

__all__ = [
    # Functions
    "hash_iterable",
    "hash_mapping",
    "hash_object",
    "hash_pandas",
    "hash_set",
]

import string
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import pandas
from pandas import DataFrame, Index, MultiIndex, Series


def hash_object(x: Any, /) -> int:
    r"""Hash an object in a permutation invariant manner."""
    if isinstance(x, Hashable):
        return hash(x)
    if isinstance(x, DataFrame | Series):
        return hash_pandas(x)
    if isinstance(x, Index | MultiIndex):
        return hash_pandas(x.to_frame())
    if isinstance(x, Mapping):
        return hash_mapping(x)
    if isinstance(x, Iterable):
        return hash_iterable(x)
    raise TypeError(f"Cannot hash object of type {type(x)}.")


def hash_set(x: Iterable[Hashable], /, *, ignore_duplicates: bool = False) -> int:
    r"""Hash an iterable of hashable objects in a permutation invariant manner."""
    if ignore_duplicates:
        return hash(frozenset(hash_object(y) for y in x))
    # We use counter as a proxy for multisets, do deal with duplicates in x.
    mdict = Counter((hash_object(y) for y in x))
    return hash(frozenset(mdict.items()))


def hash_mapping(x: Mapping[Hashable, Hashable], /) -> int:
    r"""Hash a Mapping of hashable objects in a permutation invariant manner."""
    return hash_set(x.items())


def hash_iterable(x: Iterable[Hashable], /) -> int:
    r"""Hash an iterable of hashable objects in an order dependent manner."""
    return hash_set(enumerate(x))


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


def to_base(n: int, b: int) -> list[int]:
    r"""Convert non-negative integer to any basis.

    The result satisfies: ``n = sum(d*b**k for k, d in enumerate(reversed(digits)))``

    References
    ----------
    - https://stackoverflow.com/a/28666223/9318372
    """
    digits = []
    while n:
        n, d = divmod(n, b)
        digits.append(d)
    return digits[::-1] or [0]


def to_alphanumeric(n: int) -> str:
    r"""Convert integer to alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    digits = to_base(n, len(chars))
    return "".join(chars[i] for i in digits)
