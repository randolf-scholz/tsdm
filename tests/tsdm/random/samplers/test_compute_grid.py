r"""Test compute_grid function."""

import logging
from datetime import datetime as py_dt, timedelta as py_td
from typing import Generic

import pandas as pd
from numpy import (
    datetime64 as np_dt,
    float32 as np_float,
    int32 as np_int,
    timedelta64 as np_td,
)
from pandas import Timedelta as pd_td, Timestamp as pd_dt
from pytest import mark
from typing_extensions import NamedTuple

from tsdm.random.samplers import compute_grid
from tsdm.types.time import DTVar, TDVar

__logger__ = logging.getLogger(__name__)
MODES = ["numpy", "pandas", "python", "np_int", "np_float", "int", "float"]


def _validate_grid_results(tmin, tmax, tdelta, offset):
    result = compute_grid(tmin, tmax, tdelta, offset=offset)
    kmin, kmax = result[0], result[-1]

    try:
        assert tmin <= (lower_value := offset + kmin * tdelta), f"{lower_value=}"
        assert tmin > (lower_next := offset + (kmin - 1) * tdelta), f"{lower_next=}"
        assert tmax >= (upper_value := offset + kmax * tdelta), f"{upper_value=}"
        assert tmax < (upper_next := offset + (kmax + 1) * tdelta), f"{upper_next=}"
    except AssertionError as E:
        values = {
            "tmin": tmin,
            "tmax": tmax,
            "timedelta": tdelta,
            "offset": offset,
            "kmin": kmin,
            "kmax": kmax,
        }
        raise AssertionError(f"Failed with values {values=}") from E


class GridTuple(NamedTuple, Generic[DTVar, TDVar]):
    r"""Input tuple for `compute_grid`."""

    tmin: DTVar
    tmax: DTVar
    timedelta: TDVar
    offset: TDVar


def _make_inputs(mode: str) -> GridTuple[DTVar, TDVar]:
    match mode:
        case "numpy":
            # noinspection PyArgumentList
            the_tuple = GridTuple(
                np_dt("2000-01-01"),
                np_dt("2001-01-01"),
                np_td(1, "h"),
                np_dt("2000-01-15"),
            )
        case "pandas":
            # noinspection PyArgumentList
            the_tuple = GridTuple(
                pd_dt("2000-01-01"),
                pd_dt("2001-01-01"),
                pd_td("1h"),
                pd_dt("2000-01-15"),
            )
        case "python":
            # noinspection PyArgumentList
            the_tuple = GridTuple(
                py_dt(2000, 1, 1),
                py_dt(2001, 1, 1),
                py_td(hours=1),
                py_dt(2000, 1, 15),
            )
        case "np_int":
            # noinspection PyArgumentList
            the_tuple = GridTuple(
                np_int(0),
                np_int(100),
                np_int(1),
                np_int(1),
            )
        case "np_float":
            # noinspection PyArgumentList
            the_tuple = GridTuple(
                np_float(0.0),
                np_float(99.9),
                np_float(0.6),
                np_float(1.4),
            )
        case "int":
            # noinspection PyArgumentList
            the_tuple = GridTuple(
                int(0),
                int(100),
                int(1),
                int(1),
            )
        case "float":
            # noinspection PyArgumentList
            the_tuple = GridTuple(
                float(0.0),
                float(99.9),
                float(0.6),
                float(1.4),
            )
        case _:
            raise ValueError(f"Unknown mode {mode=}")
    return the_tuple
    # return tmin, tmax, timedelta, offset  # type: ignore[return-value]


def test_edge_case() -> None:
    import math

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


@mark.parametrize("mode", MODES)
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
    tmax = tmin + pd.Timedelta(2, "h")
    timedelta = pd.Timedelta("15m")
    offset = tmin + timedelta

    _validate_grid_results(tmin, tmax, timedelta, offset)

    LOGGER.info("Finished testing on extra data")
