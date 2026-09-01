r"""Helpers for generating random arrays and time-based samples.

The functions in this module sample timestamp and timedelta grids or create
synthetic NumPy arrays for tests and examples.
"""

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

from tsdm.datatools import date_range, timedelta, timedelta_range, timestamp

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
r"""Ordered NumPy timedelta units used to choose result precision."""


def sample_timestamps(
    start: str | dt.datetime | np.datetime64 = "today",
    stop: Optional[str | dt.datetime | np.datetime64] = None,
    /,
    *,
    size: int,
    freq: str | dt.timedelta | np.timedelta64 = "1s",
    replace: bool = False,
    include_start: bool = True,
    include_final: bool = False,
) -> NDArray[np.datetime64]:
    r"""Sample timestamps from a regular grid within a time interval.

    When ``stop`` is omitted, the interval spans one day from ``start``. The
    optional boundary flags reserve the first and final grid points, respectively;
    all remaining timestamps are drawn uniformly from the interior grid.

    Args:
        start: Inclusive lower bound of the sampling interval.
        stop: Inclusive upper bound; defaults to one day after ``start``.
        size: Number of timestamps to return.
        freq: Spacing of the candidate timestamp grid.
        replace: Whether non-boundary timestamps may be selected repeatedly.
        include_start: Whether to include ``start`` in the result.
        include_final: Whether to include ``stop`` in the result.
    """
    start_dt = timestamp(start)
    final_dt = start_dt + timedelta("24h") if stop is None else timestamp(stop)
    freq_td = timedelta(freq)
    start_dt, final_dt = start_dt.round(freq_td), final_dt.round(freq_td)

    # randomly sample timestamps
    rng = np.random.default_rng()
    timestamps = date_range(start_dt, final_dt, freq=freq_td)
    np_timestamps = np.array(timestamps)

    np_timestamps = rng.choice(
        np_timestamps[include_start : -include_final or None],
        size - include_start - include_final,
        replace=replace,
    )
    np_timestamps = np.sort(np_timestamps)

    # add boundary if requested
    if include_start:
        np_timestamps = np.insert(np_timestamps, 0, start_dt)
    if include_final:
        np_timestamps = np.insert(np_timestamps, -1, final_dt)

    # Convert to base unit based on freq
    base_unit = next(u for u, val in NUMPY_TIME_UNITS.items() if freq_td >= val)
    return np_timestamps.astype(f"datetime64[{base_unit}]")


def sample_timedeltas(
    low: str | dt.timedelta | np.timedelta64 = "0s",
    high: str | dt.timedelta | np.timedelta64 = "1h",
    size: int = 1,
    /,
    *,
    freq: str | dt.timedelta | np.timedelta64 = "1s",
) -> NDArray[np.timedelta64]:
    r"""Sample timedeltas from a regularly spaced interval.

    Both bounds are rounded to ``freq`` before constructing the candidate grid.
    """
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
r"""Boolean values used when generating synthetic boolean arrays."""

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
r"""NATO-style words used when generating synthetic string arrays."""


def random_data(
    size: tuple[int], *, dtype: DTypeLike = float, missing: float = 0.0
) -> NDArray:
    r"""Create a synthetic NumPy array with values appropriate for ``dtype``.

    Integer, floating-point, boolean, and string dtypes are supported. Missing
    value generation is reserved for future implementation.
    """
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
