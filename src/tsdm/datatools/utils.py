r"""Generic data utilities."""

__all__ = [
    "date_range",
    "infer_schema",
    "timedelta",
    "timedelta_range",
    "timestamp",
    "random_partition",
]

import datetime as dt
import operator
from collections.abc import Collection, Iterable, Mapping, Sequence
from functools import wraps
from typing import Any, Optional, SupportsFloat, SupportsIndex, overload

import numpy as np
from numpy.random import Generator
from pandas import Timedelta, Timestamp
from pandas.api.typing import NaTType

from tsdm.types import HasDTypes


@overload
def random_partition[K](
    keys: Collection[K],
    *,
    sizes: Sequence[SupportsIndex],
    rng: Generator | int | None = ...,
) -> tuple[list[K], ...]: ...
@overload
def random_partition[K](
    keys: Collection[K],
    *,
    ratios: Sequence[SupportsFloat],
    rng: Generator | int | None = ...,
) -> tuple[list[K], ...]: ...
def random_partition[K](
    keys: Collection[K],
    *,
    sizes: Sequence[SupportsIndex] | None = None,
    ratios: Sequence[SupportsFloat] | None = None,
    rng: Generator | int | None = None,
) -> tuple[list[K], ...]:
    r"""Create a random partition for a dataset.

    Exactly one of `sizes` or `ratios` must be provided. Sizes are absolute
    partition counts and must sum to the number of keys. Ratios are normalized
    weights used to distribute all keys between the partitions.

    Args:
        keys: Keys to distribute among the partitions.
        sizes: Absolute partition counts. The counts must be non-negative and
            sum to the number of keys. Mutually exclusive with `ratios`.
        ratios: Relative partition weights. The weights must be non-negative
            and not all zero. Mutually exclusive with `sizes`.
        rng: Random seed or generator used to shuffle the keys.

    Returns:
        A tuple of key lists, one list for each requested partition.
    """
    if (sizes is None) == (ratios is None):
        raise ValueError("Exactly one of sizes or ratios must be provided")

    rng = np.random.default_rng(rng)
    key_map = dict(enumerate(keys))
    num = len(key_map)

    if sizes is not None:
        counts = tuple(operator.index(size) for size in sizes)
        if any(count < 0 for count in counts) or sum(counts) != num:
            raise ValueError("sizes must be non-negative and sum to the number of keys")
    else:
        assert ratios is not None
        weights = tuple(float(ratio) for ratio in ratios)

        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("ratios must be non-negative and not all zero")

        # Normalize the weights and assign leftovers to the largest remainders.
        quotas = num * np.asarray(weights) / sum(weights)
        counts = np.floor(quotas).astype(int)

        if remainder := num - counts.sum():
            idx = np.argsort(quotas - counts)
            counts[idx[-remainder:]] += 1

    indices = rng.permutation(num)
    cuts = np.cumsum(counts[:-1])

    return tuple(
        [key_map[i] for i in partition] for partition in np.split(indices, cuts)
    )


def infer_schema(table: Any, /) -> Mapping[Any, Any]:
    r"""Infer a mapping from column names to dtypes for a table-like object.

    A table's index is not included. Pass an index directly to infer its schema.

    Returns:
        A mapping from column names to dtypes.

    Raises:
        NotImplementedError: If the table type is not supported.
    """
    match table:
        case object(schema=Mapping() as schema):  # type: ignore
            # pl.DataFrame
            return dict(schema)

        case HasDTypes(dtypes=dtypes) if isinstance(dtypes, Iterable):
            # pd.MultiIndex, pd.DataFrame
            return dict(dtypes)

        case object(name=name, dtype=dtype):  # type: ignore
            # pd.Series, pd.Index, pl.Series
            return {name: dtype}

        case object(schema=schema):  # type: ignore
            # pa.Table
            return dict(zip(schema.names, schema.types, strict=True))

        case _:
            raise NotImplementedError(f"Cannot get schema for {type(table)} objects!")


@wraps(Timedelta)
def timedelta(value: Any = ..., unit: Optional[str] = None, **kwargs: Any) -> Timedelta:
    r"""Construct a `pandas.Timedelta`, raising `TypeError` if it returns `NaT`."""
    td = (
        Timedelta(unit=unit, **kwargs)
        if value is Ellipsis
        else Timedelta(value, unit=unit, **kwargs)
    )
    if isinstance(td, NaTType):
        raise TypeError("Constructor returned NaT")
    return td


@wraps(Timestamp)
def timestamp(value: Any = ..., **kwargs: Any) -> Timestamp:
    r"""Construct a `pandas.Timestamp`, raising `TypeError` if it returns `NaT`."""
    ts = Timestamp(**kwargs) if value is Ellipsis else Timestamp(value, **kwargs)
    if isinstance(ts, NaTType):
        raise TypeError("Constructor returned NaT")
    return ts


def date_range(
    start: str | dt.datetime,
    stop: str | dt.datetime,
    *,
    freq: str | dt.timedelta,
    include_end: bool = True,
) -> list[dt.datetime]:
    t0 = timestamp(start)
    t1 = timestamp(stop)
    f = timedelta(freq)
    k = (t1 - t0) // f
    items = [t0 + i * f for i in range(k)]

    if include_end and items[-1] < t1:
        items.append(t1)
    return items


def timedelta_range(
    start: str | dt.timedelta,
    stop: str | dt.timedelta,
    *,
    freq: str | dt.timedelta,
    include_end: bool = True,
) -> list[dt.timedelta]:
    t0 = timedelta(start)
    t1 = timedelta(stop)
    f = timedelta(freq)
    k = (t1 - t0) // f
    items = [t0 + i * f for i in range(k)]

    if include_end and items[-1] < t1:
        items.append(t1)
    return items
