r"""Test compute_grid function."""

import logging
import math
from datetime import datetime as py_dt, timedelta as py_td

import pandas as pd
import pytest
from numpy import (
    datetime64 as np_dt,
    float32 as np_float,
    int32 as np_int,
    timedelta64 as np_td,
)
from typing_extensions import Generic, NamedTuple

from tsdm.random.samplers import compute_grid
from tsdm.types.time import DT, TD, DateTime, TimeDelta
from tsdm.utils import timedelta as pd_td, timestamp as pd_dt

__logger__ = logging.getLogger(__name__)
MODES = ["numpy", "pandas", "python", "np_int", "np_float", "int", "float"]


def _validate_grid_results(tmin, tmax, tdelta, offset):
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


class GridTuple(NamedTuple, Generic[DT, TD]):
    r"""Input tuple for `compute_grid`."""

    tmin: DT
    tmax: DT
    offset: DT
    timedelta: TD


def _make_inputs(mode: str) -> GridTuple[DateTime, TimeDelta]:
    match mode:
        case "numpy":
            # noinspection PyArgumentList
            return GridTuple(
                tmin=np_dt("2000-01-01"),
                tmax=np_dt("2001-01-01"),
                offset=np_dt("2000-01-15"),
                timedelta=np_td(1, "h"),
            )
        case "pandas":
            # noinspection PyArgumentList
            return GridTuple(
                tmin=pd_dt("2000-01-01"),
                tmax=pd_dt("2001-01-01"),
                offset=pd_dt("2000-01-15"),
                timedelta=pd_td("1h"),
            )
        case "python":
            # noinspection PyArgumentList
            return GridTuple(
                tmin=py_dt(2000, 1, 1),
                tmax=py_dt(2001, 1, 1),
                offset=py_dt(2000, 1, 15),
                timedelta=py_td(hours=1),
            )
        case "np_int":
            # noinspection PyArgumentList
            return GridTuple(
                tmin=np_int(0),
                tmax=np_int(100),
                offset=np_int(1),
                timedelta=np_int(1),  # type: ignore[arg-type]
            )
        case "np_float":
            # noinspection PyArgumentList
            return GridTuple(
                tmin=np_float(0.0),
                tmax=np_float(99.9),
                offset=np_float(1.4),
                timedelta=np_float(0.6),  # type: ignore[arg-type]
            )
        case "int":
            # noinspection PyArgumentList
            return GridTuple(
                tmin=0,
                tmax=100,
                offset=1,
                timedelta=1,
            )
        case "float":
            # noinspection PyArgumentList
            return GridTuple(
                tmin=0.0,
                tmax=99.9,
                offset=0.6,
                timedelta=1.4,
            )
        case _:
            raise ValueError(f"Unknown mode {mode=}")


def test_edge_case() -> None:
    data = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    stride = 2
    tmin = min(data)
    tmax = max(data)
    horizon = 4
    offset = tmin + horizon
    print((tmin - offset) / stride, math.ceil((tmin - offset) / stride))  # -2.0
    print((tmax - offset) / stride, math.floor((tmax - offset) / stride))  # 2.5
    print((tmax - tmin - horizon) / stride)

    grid = compute_grid(tmin - tmin, tmax - tmin - horizon, stride)
    print(grid)


@pytest.mark.parametrize("mode", MODES)
def test_grid_pandas(mode: str) -> None:
    r"""Test compute_grid function with various input types."""
    LOGGER = __logger__.getChild(compute_grid.__name__)
    LOGGER.info("Testing mode: %s", mode)

    tmin, tmax, timedelta, offset = _make_inputs(mode)

    _validate_grid_results(tmin, tmax, timedelta, offset)

    LOGGER.info("Finished testing mode: %s", mode)


def test_grid_extra() -> None:
    r"""Test on some intervals."""
    LOGGER = __logger__.getChild(compute_grid.__name__)
    LOGGER.info("Testing on extra data")

    tmin = pd.Timestamp(0)
    tmax = tmin + pd_td(2, "h")
    td = pd_td("15m")
    offset = tmin + td

    _validate_grid_results(tmin, tmax, td, offset)

    LOGGER.info("Finished testing on extra data")
