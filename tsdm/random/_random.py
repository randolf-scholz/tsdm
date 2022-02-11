r"""Utility functions for random number generation."""

__all__ = [
    # Functions
    "sample_timestamps",
    "sample_timedeltas",
]


import logging
from typing import Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pandas import Timedelta, Timestamp, date_range, timedelta_range

from tsdm.util.dtypes import BOOLS, EMOJIS, STRINGS, TimeDeltaLike, TimeStampLike

__logger__ = logging.getLogger(__name__)


# noinspection PyTypeChecker
def sample_timestamps(
    start: TimeStampLike = "today",
    final: TimeStampLike = None,
    *,
    size: int,
    freq: TimeDeltaLike = "1s",
    replace: bool = False,
    include_start: bool = True,
    include_final: bool = False,
) -> NDArray:
    r"""Create randomly sampled timestamps.

    Parameters
    ----------
    start: TimeStampLike, default <today>
    final: TimeStampLike, default <today>+<24h>
    size: int
    freq: TimeDeltaLike, default "1s"
        The smallest possible timedelta between distinct timestamps.
    replace: bool, default True
        Whether the sample is with or without replacement.
    include_start: bool, default True
        If `True`, then `start` will always be the first sampled timestamp.
    include_final: bool, default True
        If `True`, then `final` will always be the final sampled timestamp.

    Returns
    -------
    NDArray
    """
    start = Timestamp(start)
    final = start + Timedelta("24h") if final is None else Timestamp(final)
    freq = Timedelta(freq)
    start, final = start.round(freq), final.round(freq)

    # randomly sample timestamps
    rng = np.random.default_rng()
    timestamps = date_range(start, final, freq=freq)
    timestamps = rng.choice(
        timestamps[include_start : -include_final or None],
        size - include_start - include_final,
        replace=replace,
    )
    timestamps = np.sort(timestamps)

    # add boundary if requested
    if include_start:
        timestamps = np.insert(timestamps, 0, start)
    if include_final:
        timestamps = np.insert(timestamps, -1, final)

    # Convert to base unit based on freq
    units = {
        u: np.timedelta64(1, u)
        for u in ("Y", "M", "W", "D", "h", "m", "s", "us", "ns", "ps", "fs", "as")
    }
    base_unit = next(u for u, val in units.items() if freq >= val)
    return timestamps.astype(f"datetime64[{base_unit}]")


def sample_timedeltas(
    low: Optional[TimeDeltaLike] = "0s",
    high: Optional[TimeDeltaLike] = "1h",
    size: Optional[int] = None,
    freq: Optional[TimeDeltaLike] = "1s",
) -> NDArray:
    r"""Create randomly sampled timedeltas.

    Parameters
    ----------
    low:  TimeDeltaLike, optional
    high: TimeDeltaLike, optional
    size: int,           optional
    freq: TimeDeltaLike, optional

    Returns
    -------
    NDArray
    """
    low = Timedelta(low)
    high = Timedelta(high)
    freq = Timedelta(freq)
    low, high = low.round(freq), high.round(freq)

    # randomly sample timedeltas
    rng = np.random.default_rng()
    timedeltas = timedelta_range(low, high, freq=freq)
    timedeltas = rng.choice(timedeltas, size=size)

    # Convert to base unit based on freq
    units = {
        u: np.timedelta64(1, u)
        for u in ("Y", "M", "W", "D", "h", "m", "s", "us", "ns", "ps", "fs", "as")
    }
    base_unit = next(u for u, val in units.items() if freq >= val)
    return timedeltas.astype(f"timedelta64[{base_unit}]")


def random_data(
    size: tuple[int], dtype: DTypeLike = float, missing: float = 0.5
) -> NDArray:
    r"""Create random data of given size and dtype.

    Parameters
    ----------
    size
    dtype
    missing

    Returns
    -------
    NDArray
    """
    dtype = np.dtype(dtype)
    rng = np.random.default_rng()

    if np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        data = rng.integers(low=iinfo.min, high=iinfo.max, size=size)
        result = data.astype(dtype)
    elif np.issubdtype(dtype, np.inexact):
        finfo = np.finfo(dtype)
        exp = rng.integers(low=finfo.minexp, high=finfo.maxexp, size=size)
        mant = rng.uniform(low=-2, high=+2, size=size)
        result = (mant * 2**exp).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        result = rng.choice(BOOLS, size=size)
    elif np.issubdtype(dtype, np.unicode_):
        result = rng.choice(EMOJIS, size=size)
    elif np.issubdtype(dtype, np.string_):
        result = rng.choice(STRINGS, size=size)
    else:
        raise NotImplementedError

    __logger__.warning("TODO: implement missing%s!", missing)

    return result
