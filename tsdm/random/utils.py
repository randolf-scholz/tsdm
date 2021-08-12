r"""Module Docstring."""

import logging
from typing import Final

import numpy as np
from numpy.typing import NDArray
from pandas import date_range, Timedelta, Timestamp

from tsdm.util import TimeDeltaLike, TimeStampLike

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["timesamples"]


def timesamples(
    start: TimeStampLike = "today",
    final: TimeStampLike = None,
    *,
    num: int,
    freq: TimeDeltaLike = "1s",
    replace: bool = False,
    include_start: bool = True,
    include_final: bool = False,
) -> NDArray:
    r"""Create randomly sampled timestamps.

    Parameters
    ----------
    start
    final
    num
    freq
    replace
    include_start
    include_final

    Returns
    -------
    NDArray
    """
    start = Timestamp(start)
    stop = start + Timedelta("24h") if final is None else Timestamp(final)
    freq = Timedelta(freq)
    rng = np.random.default_rng()

    time_range = date_range(start + freq, stop - freq, freq=freq)
    timestamps = rng.choice(
        time_range, num - include_start - include_final, replace=replace
    )
    if include_start:
        timestamps = np.insert(timestamps, 0, start)
    if include_final:
        timestamps = np.insert(timestamps, -1, final)

    return np.sort(timestamps)
