r"""Utility functions for random number generation."""

__all__ = [
    # Functions
    "random_data",
    "sample_timestamps",
    "sample_timedeltas",
]

from typing import Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pandas import date_range, timedelta_range

from tsdm.constants import EXAMPLE_BOOLS, EXAMPLE_EMOJIS, EXAMPLE_STRINGS, TIME_UNITS
from tsdm.types.scalars import DateTime, TimeDelta
from tsdm.utils import timedelta, timestamp


def sample_timestamps[DT: DateTime[TimeDelta]](
    start: str | DT = "today",
    final: Optional[DT] = None,
    /,
    *,
    size: int,
    freq: str | TimeDelta = "1s",
    replace: bool = False,
    include_start: bool = True,
    include_final: bool = False,
) -> NDArray:
    r"""Create randomly sampled timestamps.

    Args:
        start: TimeStampLike, default <today>
        final: TimeStampLike, default <today>+<24h>
        size: Number of timestamps to sample.
        freq: The smallest possible timedelta between distinct timestamps.
        replace: Whether the sample is with or without replacement.
        include_start: If `True`, then `start` will always be the first sampled timestamp.
        include_final: If `True`, then `final` will always be the final sampled timestamp.
    """
    start_dt = timestamp(start)
    final_dt = start_dt + timedelta("24h") if final is None else timestamp(final)
    freq_td = timedelta(freq)
    start_dt, final_dt = start_dt.round(freq_td), final_dt.round(freq_td)

    # randomly sample timestamps
    rng = np.random.default_rng()
    timestamps = date_range(start_dt, final_dt, freq=freq_td)
    timestamps = rng.choice(
        timestamps[include_start : -include_final or None],
        size - include_start - include_final,
        replace=replace,
    )
    timestamps = np.sort(timestamps)

    # add boundary if requested
    if include_start:
        timestamps = np.insert(timestamps, 0, start_dt)
    if include_final:
        timestamps = np.insert(timestamps, -1, final_dt)

    # Convert to base unit based on freq
    units: dict[str, np.timedelta64] = {
        u: np.timedelta64(1, u)
        for u in ("Y", "M", "W", "D", "h", "m", "s", "us", "ns", "ps", "fs", "as")
    }
    base_unit = next(u for u, val in units.items() if freq_td >= val)
    return timestamps.astype(f"datetime64[{base_unit}]")


def sample_timedeltas[TD: TimeDelta](
    low: str | TD = "0s",
    high: str | TD = "1h",
    size: int = 1,
    /,
    *,
    freq: str | TD = "1s",
) -> NDArray:
    r"""Create randomly sampled timedeltas."""
    low_dt = timedelta(low)
    high_dt = timedelta(high)
    freq_dt = timedelta(freq)
    low_dt, high_dt = low_dt.round(freq_dt), high_dt.round(freq_dt)

    # randomly sample timedeltas
    rng = np.random.default_rng()
    timedeltas = timedelta_range(low_dt, high_dt, freq=freq_dt)
    # convert to numpy
    base_unit = next(u for u, val in TIME_UNITS.items() if freq_dt >= val)
    numpy_timedeltas = np.asarray(timedeltas, dtype=base_unit)
    sampled_timedeltas = rng.choice(numpy_timedeltas, size=size)
    return sampled_timedeltas


def random_data(
    size: tuple[int], *, dtype: DTypeLike = float, missing: float = 0.0
) -> NDArray:
    r"""Create random data of given size and dtype."""
    if missing != 0.0:
        raise NotImplementedError("Missing values not yet implemented.")

    dtype = np.dtype(dtype)
    rng = np.random.default_rng()
    if np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        data = rng.integers(low=iinfo.min, high=iinfo.max, size=size)
        result = data.astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)  # type: ignore[arg-type]
        exp = rng.integers(low=finfo.minexp, high=finfo.maxexp, size=size)
        mant = rng.uniform(low=-2, high=+2, size=size)
        result = (mant * 2**exp).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        result = rng.choice(EXAMPLE_BOOLS, size=size)
    elif np.issubdtype(dtype, np.str_):
        result = rng.choice(EXAMPLE_EMOJIS, size=size)
    elif np.issubdtype(dtype, np.bytes_):
        result = rng.choice(EXAMPLE_STRINGS, size=size)
    else:
        raise NotImplementedError

    return result
