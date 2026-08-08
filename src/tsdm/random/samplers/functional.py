r"""Utility functions for random number generation."""

__all__ = [
    "NUMPY_TIME_UNITS",
    # Functions
    "random_data",
    "sample_timestamps",
    "sample_timedeltas",
]

import datetime as dt
from typing import Final, Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pandas import date_range, timedelta_range

from tsdm.utils import timedelta, timestamp

# NOTE: We rely on dict preserving insertion order (Python 3.7+)
NUMPY_TIME_UNITS: Final[dict[str, np.timedelta64]] = {
    "Y": np.timedelta64(1, "Y"),
    "M": np.timedelta64(1, "M"),
    "W": np.timedelta64(1, "W"),
    "D": np.timedelta64(1, "D"),
    "h": np.timedelta64(1, "h"),
    "m": np.timedelta64(1, "m"),
    "s": np.timedelta64(1, "s"),
    "us": np.timedelta64(1, "us"),
    "ns": np.timedelta64(1, "ns"),
    "ps": np.timedelta64(1, "ps"),
    "fs": np.timedelta64(1, "fs"),
    "as": np.timedelta64(1, "as"),
}
r"""Time units for `numpy.timedelta64`."""


def sample_timestamps(
    start: str | dt.datetime = "today",
    stop: Optional[dt.datetime] = None,
    /,
    *,
    size: int,
    freq: str | dt.timedelta = "1s",
    replace: bool = False,
    include_start: bool = True,
    include_final: bool = False,
) -> NDArray:
    r"""Create randomly sampled timestamps.

    Args:
        start: TimeStampLike, default <today>
        stop: TimeStampLike, default <today>+<24h>
        size: Number of timestamps to sample.
        freq: The smallest possible timedelta between distinct timestamps.
        replace: Whether the sample is with or without replacement.
        include_start: If `True`, then `start` will always be the first sampled timestamp.
        include_final: If `True`, then `final` will always be the final sampled timestamp.
    """
    start_dt = timestamp(start)
    final_dt = start_dt + timedelta("24h") if stop is None else timestamp(stop)
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
    base_unit = next(u for u, val in NUMPY_TIME_UNITS.items() if freq_td >= val)
    return timestamps.astype(f"datetime64[{base_unit}]")


def sample_timedeltas[TD: dt.timedelta](
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
    base_unit = next(u for u, val in NUMPY_TIME_UNITS.items() if freq_dt >= val)
    numpy_timedeltas = np.asarray(timedeltas, dtype=base_unit)
    sampled_timedeltas = rng.choice(numpy_timedeltas, size=size)
    return sampled_timedeltas


_EXAMPLE_BOOLS: Final[list[bool]] = [True, False]
r"""List of example bool objects."""

_EXAMPLE_STRINGS: Final[list[str]] = [
    "Alfa",
    "Bravo",
    "Charlie",
    "Delta",
    "Echo",
    "Foxtrot",
    "Golf",
    "Hotel",
    "India",
    "Juliett",
    "Kilo",
    "Lima",
    "Mike",
    "November",
    "Oscar",
    "Papa",
    "Quebec",
    "Romeo",
    "Sierra",
    "Tango",
    "Uniform",
    "Victor",
    "Whiskey",
    "X-ray",
    "Yankee",
    "Zulu",
]
r"""List of example string objects."""


def random_data(
    size: tuple[int], *, dtype: DTypeLike = float, missing: float = 0.0
) -> NDArray:
    r"""Create random data of given size and dtype."""
    if missing != 0.0:
        raise NotImplementedError("Missing values not yet implemented.")

    dtype = np.dtype(dtype)
    rng = np.random.default_rng()
    if np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)  # pyrefly: ignore[no-matching-overload]
        data = rng.integers(low=iinfo.min, high=iinfo.max, size=size)
        result = data.astype(dtype)
    elif np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)  # pyrefly: ignore[no-matching-overload]
        exp = rng.integers(low=finfo.minexp, high=finfo.maxexp, size=size)
        mant = rng.uniform(low=-2, high=+2, size=size)
        result = (mant * 2**exp).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        result = rng.choice(_EXAMPLE_BOOLS, size=size)
    elif np.issubdtype(dtype, np.str_):
        result = rng.choice(_EXAMPLE_STRINGS, size=size)
    else:
        raise NotImplementedError

    return result
