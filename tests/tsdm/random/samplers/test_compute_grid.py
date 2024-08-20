r"""Test compute_grid function."""

import logging
import math
from datetime import datetime as py_dt, timedelta as py_td
from typing import NamedTuple

import pandas as pd
import pytest
from numpy import (
    datetime64 as np_dt,
    float32 as np_float,
    int32 as np_int,
    timedelta64 as np_td,
)

from tsdm.random.samplers import compute_grid
from tsdm.types.time import DateTime, TimeDelta
from tsdm.utils import timedelta as pd_td, timestamp as pd_dt

__logger__ = logging.getLogger(__name__)


# FIXME: Use PEP 696 with python 3.13
def validate_grid_results[TD: TimeDelta](
    *, tmin: DateTime[TD], tmax: DateTime[TD], tdelta: TD, offset: DateTime[TD]
) -> None:
    result = compute_grid(tmin, tmax, tdelta, offset=offset)
    kmin, kmax = result[0], result[-1]

    lower_value = offset + kmin * tdelta
    lower_next = offset + (kmin - 1) * tdelta
    upper_value = offset + kmax * tdelta
    upper_next = offset + (kmax + 1) * tdelta

    try:
        assert tmin <= lower_value, f"{lower_value=}"
        assert tmin > lower_next, f"{lower_next=}"
        assert tmax >= upper_value, f"{upper_value=}"
        assert tmax < upper_next, f"{upper_next=}"
    except AssertionError as E:
        values = {
            "tmin": tmin,
            "tmax": tmax,
            "timedelta": tdelta,
            "offset": offset,
            "kmin": kmin,
            "kmax": kmax,
        }
        raise AssertionError(f"Failed with {values=}") from E


class GridTuple[DT: DateTime, TD: TimeDelta](NamedTuple):
    r"""Input tuple for `compute_grid`."""

    tmin: DT
    tmax: DT
    offset: DT
    timedelta: TD


EXAMPLES: dict[str, GridTuple[DateTime, TimeDelta]] = {
    "python_datetime": GridTuple(
        tmin=py_dt(2000, 1, 1),
        tmax=py_dt(2001, 1, 1),
        offset=py_dt(2000, 1, 15),
        timedelta=py_td(hours=1),
    ),
    "python_int": GridTuple(
        tmin=0,
        tmax=100,
        offset=1,
        timedelta=1,
    ),
    "python_float": GridTuple(
        tmin=0.0,
        tmax=99.9,
        offset=0.6,
        timedelta=1.4,
    ),
    "pandas_datetime": GridTuple(
        tmin=pd_dt("2000-01-01"),
        tmax=pd_dt("2001-01-01"),
        offset=pd_dt("2000-01-15"),
        timedelta=pd_td("1h"),
    ),
    "numpy_datetime": GridTuple(
        tmin=np_dt("2000-01-01"),
        tmax=np_dt("2001-01-01"),
        offset=np_dt("2000-01-15"),
        timedelta=np_td(1, "h"),
    ),
    "numpy_float": GridTuple(
        tmin=np_float(0.0),
        tmax=np_float(99.9),
        offset=np_float(1.4),
        timedelta=np_float(0.6),  # type: ignore[arg-type]
    ),
    "numpy_int": GridTuple(
        tmin=np_int(0),
        tmax=np_int(100),
        offset=np_int(1),
        timedelta=np_int(1),  # type: ignore[arg-type]
    ),
}


def test_edge_case() -> None:
    r"""Test edge case.

    In this example, The following windows are considerable:
    [11, 12, 13, 14]  (initial window)
    [13, 14, 15, 16]  (shifted once)
    [15, 16, 17, 18]  (shifted twice)
    [17, 18, 19, 20]  (shifted thrice)

    However, the last window is excluded, despite being within the range.
    This is because compute grid checks for the true inequality `t_last < t_max`.

    This is the correct thing to do for continuous data, but not for discrete data.
    """
    data = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    stride = 2
    tmin = min(data)
    tmax = max(data)
    horizon = 4
    offset = tmin + horizon
    assert (tmin - offset) / stride == -2.0
    assert math.ceil((tmin - offset) / stride) == -2
    assert (tmax - offset) / stride == +2.5
    assert math.floor((tmax - offset) / stride) == +2
    assert (tmax - tmin - horizon) / stride == 2.5
    grid = compute_grid(tmin - tmin, tmax - tmin - horizon, stride)
    assert grid == [0, 1, 2]


@pytest.mark.parametrize("name", EXAMPLES)
def test_grid_pandas(name: str) -> None:
    r"""Test compute_grid function with various input types."""
    inputs = EXAMPLES[name]
    validate_grid_results(
        tmin=inputs.tmin,
        tmax=inputs.tmax,
        tdelta=inputs.timedelta,
        offset=inputs.offset,
    )


def test_grid_extra() -> None:
    r"""Test on some intervals."""
    LOGGER = __logger__.getChild(compute_grid.__name__)
    LOGGER.info("Testing on extra data")

    tmin = pd.Timestamp(0)
    tmax = tmin + pd_td(2, "h")
    td = pd_td("15m")
    offset = tmin + td

    validate_grid_results(tmin=tmin, tmax=tmax, tdelta=td, offset=offset)

    LOGGER.info("Finished testing on extra data")
