r"""Generic data utilities."""

__all__ = [
    "Unit",
    "date_range",
    "timedelta",
    "timedelta_range",
    "timestamp",
    "to_timedelta",
]


import datetime as dt
from enum import StrEnum
from functools import wraps
from typing import Any, SupportsFloat, SupportsIndex, overload

import numpy as np
from pandas import Timedelta, Timestamp
from pandas.api.typing import NaTType

_MISSING = sentinel("_MISSING")


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


class Unit(StrEnum):
    r"""Time unit."""

    WEEKS = "weeks"
    DAYS = "days"
    HOURS = "hours"
    MINUTES = "minutes"
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    NANOSECONDS = "nanoseconds"

    @classmethod
    def _missing_(cls, value: object) -> Unit | None:
        if not isinstance(value, str):
            return None
        return _UNIT_MAPPING.get(value.lower())


_UNIT_MAPPING = {
    **dict.fromkeys(("w", "wk", "week", "weeks"), Unit.WEEKS),
    **dict.fromkeys(("d", "day", "days"), Unit.DAYS),
    **dict.fromkeys(("h", "hr", "hrs", "hour", "hours"), Unit.HOURS),
    **dict.fromkeys(("m", "min", "mins", "minute", "minutes"), Unit.MINUTES),
    **dict.fromkeys(("s", "sec", "secs", "second", "seconds"), Unit.SECONDS),
    **dict.fromkeys(("ms", "mili", "milis", "millisecond", "milliseconds"), Unit.MILLISECONDS),
    **dict.fromkeys(("us", "µs", "μs", "microsecond", "microseconds"), Unit.MICROSECONDS),
    **dict.fromkeys(("ns", "nano", "nanos", "nanosecond", "nanoseconds"), Unit.NANOSECONDS),
}  # fmt: skip


def _parse_string_value(value: str, /) -> dt.timedelta | None:
    raise NotImplementedError


# fmt: off
@overload
def to_timedelta[T: None | dt.timedelta](value: T, /) -> T: ...
@overload
def to_timedelta(value: np.timedelta64 | str, /) -> dt.timedelta | None: ...
@overload
def to_timedelta(value: SupportsFloat | SupportsIndex, /, *, unit: str) -> dt.timedelta | None: ...
# fmt: on
@overload
def to_timedelta(
    *,
    weeks: float = 0,
    days: float = 0,
    hours: float = 0,
    minutes: float = 0,
    seconds: float = 0,
    milliseconds: float = 0,
    microseconds: float = 0,
    nanoseconds: float = 0,
) -> dt.timedelta | None: ...
def to_timedelta(
    value: object = _MISSING, /, *, unit: str | None = None, **kwargs: float
) -> dt.timedelta | None:
    r"""Convert object to timedelta."""
    # last overload
    if value is _MISSING:
        if unit is not None:
            raise ValueError("unit must be None if value is not provided")
        if kwargs.pop("nanoseconds") != 0:
            raise ValueError(
                "nanoseconds not supported, "
                "see: https://github.com/python/cpython/pull/92078"
            )
        return dt.timedelta(**kwargs)
    if kwargs:
        raise ValueError("kwargs must be empty if value is provided")

    # first overload
    if value is None:
        return value
    if isinstance(value, dt.timedelta):
        return value

    # second overload
    if isinstance(value, np.timedelta64):
        delta = value.item()
        if delta is None or isinstance(delta, dt.timedelta):
            return delta
        raise ValueError(f"Cannot convert {value} to dt.timedelta (maybe")
    if isinstance(value, str):
        if unit is not None:
            raise ValueError("unit must be None if value is a string")
        # parse string, e.g. "1 day", "2h30m", "17.4us"
        return _parse_string_value(value)

    # 3rd overload
    if isinstance(value, SupportsIndex | SupportsFloat):
        if unit is None:
            raise ValueError("unit must be provided if value is a number")
        try:
            delta = value.__index__()  # type: ignore
        except Exception:
            delta = value.__float__()  # type: ignore
        unit = Unit(unit)
        return to_timedelta(**{unit: delta})

    # all overloads exhausted
    raise ValueError(f"Cannot convert {value} to dt.timedelta")
