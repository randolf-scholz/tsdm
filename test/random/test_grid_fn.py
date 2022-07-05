#!/usr/bin/env python
r"""Test grid function."""

import logging
from datetime import datetime as py_dt
from datetime import timedelta as py_td
from typing import Any

import pandas as pd
from numpy import datetime64 as np_dt
from numpy import float32 as np_float
from numpy import int32 as np_int
from numpy import timedelta64 as np_td
from pandas import Timedelta as pd_td
from pandas import Timestamp as pd_dt
from pytest import mark

from tsdm.random.samplers import grid

__logger__ = logging.getLogger(__name__)
MODES = ["numpy", "pandas", "python", "np_int", "np_float", "int", "float"]


def _validate_grid_results(tmin, tmax, timedelta, offset):
    result = grid(tmin, tmax, timedelta, offset=offset)
    kmin, kmax = result[0], result[-1]

    try:
        lower_bound = offset + kmin * timedelta
        upper_bound = offset + kmax * timedelta
        lower_break = offset + (kmin - 1) * timedelta
        upper_break = offset + (kmax + 1) * timedelta
        assert tmin <= lower_bound, f"{lower_bound=}"
        assert tmin > lower_break, f"{lower_break=}"
        assert tmax >= upper_bound, f"{upper_bound=}"
        assert tmax < upper_break, f"{upper_break=}"
    except AssertionError as E:
        values = {
            "tmin": tmin,
            "tmax": tmax,
            "timedelta": timedelta,
            "offset": offset,
            "kmin": kmin,
            "kmax": kmax,
        }
        raise AssertionError(f"Failed with values {values=}") from E


@mark.parametrize("mode", MODES)
def test_grid_pandas(mode):
    r"""Test grid function with various input types."""
    __logger__.info("Testing  grid with mode: %s", mode)

    tmin: Any
    tmax: Any
    timedelta: Any

    if mode == "numpy":
        tmin = np_dt("2000-01-01")
        tmax = np_dt("2001-01-01")
        timedelta = np_td(1, "h")
        offset = np_dt("2000-01-15")
    elif mode == "pandas":
        tmin = pd_dt("2000-01-01")
        tmax = pd_dt("2001-01-01")
        timedelta = pd_td("1h")
        offset = pd_dt("2000-01-15")
    elif mode == "python":
        tmin = py_dt(2000, 1, 1)
        tmax = py_dt(2001, 1, 1)
        timedelta = py_td(hours=1)
        offset = py_dt(2000, 1, 15)
    elif mode == "np_int":
        tmin = np_int(0)
        tmax = np_int(100)
        timedelta = np_int(1)
        offset = np_int(1)
    elif mode == "np_float":
        tmin = np_float(0.0)
        tmax = np_float(99.9)
        timedelta = np_float(0.6)
        offset = np_float(1.4)
    elif mode == "int":
        tmin = int(0)
        tmax = int(100)
        timedelta = int(1)
        offset = int(1)
    elif mode == "float":
        tmin = float(0.0)
        tmax = float(99.9)
        timedelta = float(0.6)
        offset = float(1.4)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _validate_grid_results(tmin, tmax, timedelta, offset)

    __logger__.info("Finished grid with mode: %s", mode)


def test_grid_extra():
    r"""Test on some intervals."""
    __logger__.info("Testing  grid on extra data")

    tmin = pd.Timestamp(0)
    tmax = tmin + pd.Timedelta(2, "h")
    timedelta = pd.Timedelta("15m")
    offset = tmin + timedelta

    _validate_grid_results(tmin, tmax, timedelta, offset)

    __logger__.info("Finished grid on extra data")


def __main__():
    logging.basicConfig(level=logging.INFO)
    for mode in MODES:
        test_grid_pandas(mode)
    test_grid_extra()


if __name__ == "__main__":
    __main__()
