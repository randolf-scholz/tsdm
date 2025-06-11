r"""Tests for time related types."""

from datetime import datetime as py_datetime, timedelta as py_timedelta
from typing import Literal, assert_type

import numpy as np
import pandas as pd
import pytest

from tsdm.testing import check_shared_interface, supports_issubclass
from tsdm.types.scalars import BoolScalar, DurationScalar, TimestampScalar
from tsdm.utils import timedelta, timestamp

# region setup -------------------------------------------------------------------------
ISO_DATE = "2021-01-01"
# python timedeltas
TD_PY_FLOAT: float = 10.0
TD_PY_INT: int = 10
TD_PY_DUR: py_timedelta = py_timedelta(days=1)
# numpy timedeltas
TD_NP_FLOAT: np.float64 = np.float64(10.0)
TD_NP_INT: np.int64 = np.int64(10)
TD_NP_DUR: "np.timedelta64[py_timedelta]" = np.timedelta64(1, "D")
# pandas timedeltas
TD_PD_DUR: pd.Timedelta = timedelta(days=1)
# python timestamps
TS_PY_DATE: py_datetime = py_datetime.fromisoformat(ISO_DATE)
TS_PY_FLOAT: float = 10.0
TS_PY_INT: int = 10
# numpy timestamps
TS_NP_DATE: "np.datetime64[py_datetime]" = np.datetime64(ISO_DATE)
TS_NP_FLOAT: np.float64 = np.float64(TS_PY_FLOAT)
TS_NP_INT: np.int64 = np.int64(TS_PY_INT)
# pandas timestamps
TS_PD_DATE: pd.Timestamp = timestamp(ISO_DATE)

type PY_TD = Literal["python[float]", "python[int]", "python[timedelta]"]
type PY_TS = Literal["python[float]", "python[int]", "python[datetime]"]
type NP_TD = Literal["numpy[float]", "numpy[int]", "numpy[timedelta]"]
type NP_TS = Literal["numpy[float]", "numpy[int]", "numpy[datetime]"]
type PD_TD = Literal["pandas[timedelta]"]
type PD_TS = Literal["pandas[datetime]"]
type TD = PY_TD | NP_TD | PD_TD
type TS = PY_TS | NP_TS | PD_TS
# endregion setup ----------------------------------------------------------------------

# region test data ---------------------------------------------------------------------
TIMEDELTAS: dict[TD, DurationScalar] = {
    "numpy[float]"      : TD_NP_FLOAT,
    "numpy[int]"        : TD_NP_INT,
    "numpy[timedelta]"  : TD_NP_DUR,
    "pandas[timedelta]" : TD_PD_DUR,
    "python[float]"     : TD_PY_FLOAT,
    "python[int]"       : TD_PY_INT,
    "python[timedelta]" : TD_PY_DUR,
}  # fmt: skip
r"""Dictionary of timedelta scalars."""

DURATION_TIMEDELTAS: dict[TD, DurationScalar] = {
    "numpy[timedelta]"  : TD_NP_DUR,
    "pandas[timedelta]" : TD_PD_DUR,
    "python[timedelta]" : TD_PY_DUR,
}  # fmt: skip
r"""Dictionary of timedelta-like durations."""

TIMESTAMPS: dict[TS, TimestampScalar] = {
    "numpy[datetime]"  : TS_NP_DATE,
    "numpy[float]"     : TS_NP_FLOAT,
    "numpy[int]"       : TS_NP_INT,
    "pandas[datetime]" : TS_PD_DATE,
    "python[datetime]" : TS_PY_DATE,
    "python[float]"    : TS_PY_FLOAT,
    "python[int]"      : TS_PY_INT,
}  # fmt: skip
r"""Dictionary of timestamp scalars."""

DATE_TIMESTAMPS: dict[TS, TimestampScalar[py_timedelta]] = {
    "numpy[datetime]"  : TS_NP_DATE,
    "pandas[datetime]" : TS_PD_DATE,
    "python[datetime]" : TS_PY_DATE,
}  # fmt: skip
r"""Dictionary of datetime-like timestamps."""

FLOAT_TIMESTAMPS: dict[TS, TimestampScalar[float]] = {
    "numpy[float]"  : TS_NP_FLOAT,
    "python[float]" : TS_PY_FLOAT,
}  # fmt: skip
r"""Dictionary of float-like timestamps."""

INT_TIMESTAMPS: dict[TS, TimestampScalar[int]] = {
    "numpy[int]"  : TS_NP_INT,
    "python[int]" : TS_PY_INT,
}  # fmt: skip
r"""Dictionary of int-like timestamps."""
# endregion test data ------------------------------------------------------------------


def test_assign() -> None:
    r"""Test the datetime protocol."""
    _0: DurationScalar = TD_NP_FLOAT
    _1: DurationScalar = TD_NP_INT
    _2: DurationScalar = TD_NP_DUR
    _3: DurationScalar = TD_PD_DUR
    _4: DurationScalar = TD_PY_FLOAT
    _5: DurationScalar = TD_PY_INT
    _6: DurationScalar = TD_PY_DUR


def test_timestamp_issubclass() -> None:
    r"""Test the datetime protocol."""
    assert supports_issubclass(TimestampScalar)


def test_timedelta_issubclass() -> None:
    r"""Test the datetime protocol."""
    assert supports_issubclass(DurationScalar)


def test_joint_attrs_datetime() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(
        DATE_TIMESTAMPS.values(), TimestampScalar, raise_on_extra=False
    )


def test_joint_attrs_timestamp() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(TIMESTAMPS.values(), TimestampScalar, raise_on_extra=False)


def test_joint_attrs_timedelta() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(TIMEDELTAS.values(), DurationScalar, raise_on_extra=False)


@pytest.mark.parametrize("name", TIMESTAMPS)
def test_timestamp_protocol(name: TS) -> None:
    r"""Test the datetime protocol."""
    TS_value = TIMESTAMPS[name]
    assert isinstance(TS_value, TimestampScalar)
    assert issubclass(type(TS_value), TimestampScalar)

    # test __sub__
    zero = TS_value - TS_value
    assert isinstance(zero, DurationScalar)
    assert issubclass(type(zero), DurationScalar)

    # test __add__
    TS_new = TS_value + zero
    assert isinstance(TS_new, TimestampScalar)
    assert issubclass(type(TS_new), TimestampScalar)

    # test __ge__
    result = TS_value >= TS_value
    assert result
    assert isinstance(result, BoolScalar)


@pytest.mark.parametrize("name", TIMEDELTAS)
def test_timedelta_protocol(name: TD) -> None:
    r"""Test the datetime protocol."""
    td_value = TIMEDELTAS[name]
    original_type = type(td_value)
    assert isinstance(td_value, DurationScalar)
    assert issubclass(original_type, DurationScalar)

    # test __ge__
    result_ge = td_value >= td_value
    assert result_ge
    assert isinstance(result_ge, BoolScalar)
    assert issubclass(type(result_ge), BoolScalar)

    # test __pos__
    result_pos = +td_value
    assert type(result_pos) is original_type

    # test __neg__
    result_neg = -td_value
    assert type(result_neg) is original_type

    # test __add__
    result_add = td_value + td_value
    assert type(result_add) is original_type

    # test __sub__
    result_sub = td_value - td_value
    assert type(result_sub) is original_type

    # test __mul__
    result_mul_int = td_value * 2
    assert type(result_mul_int) is original_type

    # test __floordiv__
    result_fdiv = td_value // 2
    assert type(result_fdiv) is original_type

    # test __truediv__ with self
    result_div_self = td_value / td_value
    assert isinstance(result_div_self, float)

    # test __truediv__
    result_div_int = td_value / 2
    assert isinstance(result_div_int, DurationScalar)


def test_timestamp_assign() -> None:
    TS_float: TimestampScalar[float] = TS_PY_FLOAT
    TS_int: TimestampScalar[int] = TS_PY_INT
    TS_numpy: TimestampScalar[np.timedelta64[py_timedelta]] = TS_NP_DATE
    TS_numpy_float: TimestampScalar[np.float64] = TS_NP_FLOAT
    TS_numpy_int: TimestampScalar[np.int64] = TS_NP_INT
    TS_pandas: TimestampScalar[pd.Timedelta] = TS_PD_DATE
    TS_python: TimestampScalar[py_timedelta] = TS_PY_DATE

    assert isinstance(TS_float, TimestampScalar)
    assert isinstance(TS_int, TimestampScalar)
    assert isinstance(TS_numpy, TimestampScalar)
    assert isinstance(TS_numpy_float, TimestampScalar)
    assert isinstance(TS_numpy_int, TimestampScalar)
    assert isinstance(TS_pandas, TimestampScalar)
    assert isinstance(TS_python, TimestampScalar)


def test_timedelta_assign() -> None:
    td_float: DurationScalar = TD_PY_FLOAT
    td_int: DurationScalar = TD_PY_INT
    td_numpy: DurationScalar = TD_NP_DUR
    td_numpy_float: DurationScalar = TD_NP_FLOAT
    td_numpy_int: DurationScalar = TD_NP_INT
    td_pandas: DurationScalar = TD_PD_DUR
    td_python: DurationScalar = TD_PY_DUR

    assert isinstance(td_float, DurationScalar)
    assert isinstance(td_int, DurationScalar)
    assert isinstance(td_numpy, DurationScalar)
    assert isinstance(td_numpy_float, DurationScalar)
    assert isinstance(td_numpy_int, DurationScalar)
    assert isinstance(td_pandas, DurationScalar)
    assert isinstance(td_python, DurationScalar)


def test_timestamp_typevar() -> None:
    r"""Type-Checking TS_VAR."""

    def id_dt[DT: TimestampScalar](x: DT, /) -> DT:
        return x

    id_dt(TS_PY_FLOAT)
    id_dt(TS_PY_INT)
    id_dt(TS_NP_DATE)
    id_dt(TS_NP_FLOAT)
    id_dt(TS_NP_INT)
    id_dt(TS_PD_DATE)
    id_dt(TS_PY_DATE)


def test_timestamp_difference() -> None:
    r"""Test inference capabilities of type checkers."""

    def infer_delta_type[TD: DurationScalar](x: TimestampScalar[TD]) -> TD:
        return x - x

    assert_type(TS_PY_FLOAT - TS_PY_FLOAT, float)
    assert_type(TS_PY_INT - TS_PY_INT, int)
    assert_type(TS_NP_DATE - TS_NP_DATE, np.timedelta64)
    assert_type(TS_NP_FLOAT - TS_NP_FLOAT, np.float64)
    assert_type(TS_NP_INT - TS_NP_INT, np.int64)
    assert_type(TS_PD_DATE - TS_PD_DATE, pd.Timedelta)
    assert_type(TS_PY_DATE - TS_PY_DATE, py_timedelta)

    assert_type(infer_delta_type(TS_PY_FLOAT), float)
    assert_type(infer_delta_type(TS_PY_INT), int)
    assert_type(infer_delta_type(TS_NP_DATE), np.timedelta64)
    assert_type(infer_delta_type(TS_NP_FLOAT), np.float64)
    assert_type(infer_delta_type(TS_NP_INT), np.int64)
    assert_type(infer_delta_type(TS_PD_DATE), pd.Timedelta)
    assert_type(infer_delta_type(TS_PY_DATE), py_timedelta)


def test_td_var() -> None:
    r"""Type-Checking TD_VAR."""

    def id_td[TD: DurationScalar](x: TD, /) -> TD:
        return x

    id_td(TD_PY_FLOAT)
    id_td(TD_PY_INT)
    id_td(TD_NP_DUR)
    id_td(TD_NP_FLOAT)
    id_td(TD_NP_INT)
    id_td(TD_PD_DUR)
    id_td(TD_PY_DUR)
