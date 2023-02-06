#!/usr/bin/env python
r"""Test compute_grid function."""

import logging
from datetime import datetime as py_dt
from datetime import timedelta as py_td
from typing import Generic

import pandas as pd
from numpy import datetime64 as np_dt
from numpy import float32 as np_float
from numpy import int32 as np_int
from numpy import timedelta64 as np_td
from pandas import Timedelta as pd_td
from pandas import Timestamp as pd_dt
from pytest import mark
from typing_extensions import NamedTuple  # FIXME: remove with python 3.11

from tsdm.random.samplers import compute_grid
from tsdm.types.time import DTVar, TDVar

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)
MODES = ["numpy", "pandas", "python", "np_int", "np_float", "int", "float"]


def _validate_grid_results(tmin, tmax, timedelta, offset):
    result = compute_grid(tmin, tmax, timedelta, offset=offset)
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


@mark.parametrize("mode", MODES)
def test_grid_pandas(mode):
    r"""Test compute_grid function with various input types."""
    LOGGER = __logger__.getChild(compute_grid.__name__)
    LOGGER.info("Testing mode: %s", mode)

    tmin, tmax, timedelta, offset = _make_inputs(mode)

    _validate_grid_results(tmin, tmax, timedelta, offset)

    LOGGER.info("Finished testing mode: %s", mode)


def test_grid_extra():
    r"""Test on some intervals."""
    LOGGER = __logger__.getChild(compute_grid.__name__)
    LOGGER.info("Testing on extra data")

    tmin = pd.Timestamp(0)
    tmax = tmin + pd.Timedelta(2, "h")
    timedelta = pd.Timedelta("15m")
    offset = tmin + timedelta

    _validate_grid_results(tmin, tmax, timedelta, offset)

    LOGGER.info("Finished testing on extra data")


def _main() -> None:
    for mode in MODES:
        test_grid_pandas(mode)
    test_grid_extra()


if __name__ == "__main__":
    _main()
