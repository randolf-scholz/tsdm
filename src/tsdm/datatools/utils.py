r"""Generic data utilities."""

__all__ = [
    "date_range",
    "timedelta",
    "timedelta_range",
    "timestamp",
]

import datetime as dt
from functools import wraps
from typing import Any

from pandas import Timedelta, Timestamp
from pandas.api.typing import NaTType


@wraps(Timedelta)
def timedelta(value: Any = ..., unit: str | None = None, **kwargs: Any) -> Timedelta:
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
