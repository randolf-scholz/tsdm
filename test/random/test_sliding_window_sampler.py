#!/usr/bin/env python
r"""Test grid function."""

import logging
from datetime import datetime as py_dt
from datetime import timedelta as py_td
from typing import Any

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


@mark.parametrize("mode", MODES)
def test_grid_pandas(mode):
    r"""Test grid function with various input types."""
    __logger__.info("Testing  grid with mode: %s", mode)

    tmin: Any
    tmax: Any
    timedelta: Any

    if mode == "numpy":
        tmin = np_dt("2000-01-01")
        tmax = np_dt("2000-01-02")
        timedelta = np_td(1, "h")
    elif mode == "pandas":
        tmin = pd_dt("2000-01-01")
        tmax = pd_dt("2000-01-02")
        timedelta = pd_td("1h")
    elif mode == "python":
        tmin = py_dt(2000, 1, 1)
        tmax = py_dt(2000, 1, 2)
        timedelta = py_td(hours=1)
    elif mode == "np_int":
        tmin = np_int(2000)
        tmax = np_int(2001)
        timedelta = np_int(1)
    elif mode == "np_float":
        tmin = np_float(2000)
        tmax = np_float(2001)
        timedelta = np_float(1)
    elif mode == "int":
        tmin = int(2000)
        tmax = int(2001)
        timedelta = int(1)
    elif mode == "float":
        tmin = float(2000)
        tmax = float(2001)
        timedelta = float(1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    grid(tmin, tmax, timedelta)

    __logger__.info("Finished grid with mode: %s", mode)


def __main__():
    logging.basicConfig(level=logging.INFO)
    for mode in MODES:
        test_grid_pandas(mode)


if __name__ == "__main__":
    __main__()
