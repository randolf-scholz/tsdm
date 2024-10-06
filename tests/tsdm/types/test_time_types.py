r"""Tests for time related types."""

from datetime import datetime as python_datetime, timedelta as python_timedelta
from typing import assert_type

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.typing import NDArray
from typing_extensions import get_protocol_members

from tsdm.testing import check_shared_interface
from tsdm.types.scalars import BoolScalar, DateTime, TimeDelta
from tsdm.utils import timedelta, timestamp

ISO_DATE = "2021-01-01"


# region scalar datetimes --------------------------------------------------------------
DT_FLOAT: float = 10.0
DT_INT: int = 10
DT_NUMPY: np.datetime64 = np.datetime64(ISO_DATE)
DT_NUMPY_FLOAT: np.float64 = np.float64(DT_FLOAT)
DT_NUMPY_INT: np.int64 = np.int64(DT_INT)
DT_PANDAS: pd.Timestamp = timestamp(ISO_DATE)
DT_PYTHON: python_datetime = python_datetime.fromisoformat(ISO_DATE)
DT_ARROW = pa.scalar(DT_PYTHON, type=pa.timestamp("ms"))
DATETIMES: dict[str, DateTime] = {
    "float"       : DT_FLOAT,
    "int"         : DT_INT,
    "numpy"       : DT_NUMPY,
    "numpy_float" : DT_NUMPY_FLOAT,
    "numpy_int"   : DT_NUMPY_INT,
    "pandas"      : DT_PANDAS,
    "python"      : DT_PYTHON,
    # NOT SUPPORTED: "arrow"       : DT_ARROW,
}  # fmt: skip
# endregion scalar datetimes -----------------------------------------------------------

# region scalar timedeltas -------------------------------------------------------------
TD_FLOAT: float = 10.0
TD_INT: int = 10
TD_NUMPY: np.timedelta64 = np.timedelta64(1, "D")
TD_NUMPY_FLOAT: np.float64 = np.float64(10.0)
TD_NUMPY_INT: np.int64 = np.int64(10)
TD_PANDAS: pd.Timedelta = timedelta(days=1)
TD_PYTHON: python_timedelta = python_timedelta(days=1)
TD_ARROW = pa.scalar(10, type=pa.duration("ms"))
TIMEDELTAS: dict[str, TimeDelta] = {
    "python_float"     : TD_FLOAT,
    "python_int"       : TD_INT,
    "numpy_timedelta"  : TD_NUMPY,
    "numpy_float"      : TD_NUMPY_FLOAT,
    "numpy_int"        : TD_NUMPY_INT,
    "pandas_timedelta" : TD_PANDAS,
    "python_timedelta" : TD_PYTHON,
    # NOT SUPPORTED: "arrow"       : TD_ARROW,
}  # fmt: skip
# endregion scalar timedeltas ----------------------------------------------------------

# region array datetimes ---------------------------------------------------------------
DT_NDARRAY: NDArray[np.datetime64] = np.array([DT_NUMPY])
DT_NDARRAY_FLOAT: NDArray[np.float64] = np.array([DT_NUMPY_FLOAT])
DT_NDARRAY_INT: NDArray[np.int64] = np.array([DT_NUMPY_INT])
# endregion array datetimes ------------------------------------------------------------

# region array timedeltas --------------------------------------------------------------
TD_NDARRAY: NDArray[np.timedelta64] = np.array([TD_NUMPY])
TD_NDARRAY_FLOAT: NDArray[np.float64] = np.array([TD_NUMPY_FLOAT])
TD_NDARRAY_INT: NDArray[np.int64] = np.array([TD_NUMPY_INT])
# endregion array timedeltas -----------------------------------------------------------


def test_datetime_protocol_itself() -> None:
    r"""Test the datetime protocol."""
    non_callable_members = {
        member
        for member in get_protocol_members(DateTime)
        if not callable(getattr(DateTime, member))
    }
    assert not non_callable_members
    assert issubclass(DateTime, DateTime)


def test_timedelta_protocol_itself() -> None:
    r"""Test the datetime protocol."""
    non_callable_members = {
        member
        for member in get_protocol_members(TimeDelta)
        if not callable(getattr(TimeDelta, member))
    }
    assert not non_callable_members
    assert issubclass(TimeDelta, TimeDelta)


@pytest.mark.parametrize("name", DATETIMES)
def test_datetime_protocol(name: str) -> None:
    r"""Test the datetime protocol."""
    dt_value = DATETIMES[name]
    assert isinstance(dt_value, DateTime)
    assert issubclass(type(dt_value), DateTime)

    # test __sub__
    zero = dt_value - dt_value
    assert isinstance(zero, TimeDelta)
    assert issubclass(type(zero), TimeDelta)

    # test __add__
    dt_new = dt_value + zero
    assert isinstance(dt_new, DateTime)
    assert issubclass(type(dt_new), DateTime)

    # test __ge__
    result = dt_value >= dt_value
    assert result
    assert isinstance(result, BoolScalar)


@pytest.mark.parametrize("name", TIMEDELTAS)
def test_timedelta_protocol(name: str) -> None:
    r"""Test the datetime protocol."""
    td_value = TIMEDELTAS[name]
    original_type = type(td_value)
    assert isinstance(td_value, TimeDelta)
    assert issubclass(original_type, TimeDelta)

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
    assert isinstance(result_div_int, TimeDelta)


def test_joint_attrs_datetime() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(DATETIMES.values(), DateTime, raise_on_extra=False)


def test_joint_attrs_timedelta() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(TIMEDELTAS.values(), TimeDelta, raise_on_extra=False)


def test_datetime_assign() -> None:
    dt_float: DateTime[float] = DT_FLOAT
    dt_int: DateTime[int] = DT_INT
    dt_numpy: DateTime[np.timedelta64] = DT_NUMPY
    dt_numpy_float: DateTime[np.float64] = DT_NUMPY_FLOAT
    dt_numpy_int: DateTime[np.int64] = DT_NUMPY_INT
    dt_pandas: DateTime[pd.Timedelta] = DT_PANDAS
    dt_python: DateTime[python_timedelta] = DT_PYTHON

    assert isinstance(dt_float, DateTime)
    assert isinstance(dt_int, DateTime)
    assert isinstance(dt_numpy, DateTime)
    assert isinstance(dt_numpy_float, DateTime)
    assert isinstance(dt_numpy_int, DateTime)
    assert isinstance(dt_pandas, DateTime)
    assert isinstance(dt_python, DateTime)


def test_timedelta_assign() -> None:
    td_float: TimeDelta = TD_FLOAT
    td_int: TimeDelta = TD_INT
    td_numpy: TimeDelta = TD_NUMPY
    td_numpy_float: TimeDelta = TD_NUMPY_FLOAT
    td_numpy_int: TimeDelta = TD_NUMPY_INT
    td_pandas: TimeDelta = TD_PANDAS
    td_python: TimeDelta = TD_PYTHON

    assert isinstance(td_float, TimeDelta)
    assert isinstance(td_int, TimeDelta)
    assert isinstance(td_numpy, TimeDelta)
    assert isinstance(td_numpy_float, TimeDelta)
    assert isinstance(td_numpy_int, TimeDelta)
    assert isinstance(td_pandas, TimeDelta)
    assert isinstance(td_python, TimeDelta)


def test_dt_var() -> None:
    r"""Type-Checking DT_VAR."""

    def id_dt[DT: DateTime](x: DT, /) -> DT:
        return x

    id_dt(DT_FLOAT)
    id_dt(DT_INT)
    id_dt(DT_NUMPY)
    id_dt(DT_NUMPY_FLOAT)
    id_dt(DT_NUMPY_INT)
    id_dt(DT_PANDAS)
    id_dt(DT_PYTHON)


def test_dt_diff() -> None:
    r"""Test inference capabilities of type checkers."""

    def diff[TD: TimeDelta](x: DateTime[TD]) -> TD:
        return x - x

    assert_type(diff(DT_FLOAT), float)
    assert_type(diff(DT_INT), int)
    assert_type(diff(DT_NUMPY), np.timedelta64)
    assert_type(diff(DT_NUMPY_FLOAT), np.float64)  # type: ignore[assert-type, misc]
    assert_type(diff(DT_NUMPY_INT), np.float32)  # type: ignore[assert-type, misc]
    assert_type(diff(DT_PANDAS), pd.Timedelta)
    assert_type(diff(DT_PYTHON), python_timedelta)


def test_td_var() -> None:
    r"""Type-Checking TD_VAR."""

    def id_td[TD: TimeDelta](x: TD, /) -> TD:
        return x

    id_td(TD_FLOAT)
    id_td(TD_INT)
    id_td(TD_NUMPY)
    id_td(TD_NUMPY_FLOAT)
    id_td(TD_NUMPY_INT)
    id_td(TD_PANDAS)
    id_td(TD_PYTHON)
