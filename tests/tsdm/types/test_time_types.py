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
from tsdm.types.scalars import BoolScalar, TimeDelta, TimeStamp
from tsdm.utils import timedelta, timestamp

ISO_DATE = "2021-01-01"


# region scalar datetimes --------------------------------------------------------------
TS_PY_DATE: python_datetime = python_datetime.fromisoformat(ISO_DATE)
TS_PY_FLOAT: float = 10.0
TS_PY_INT: int = 10
TS_NP_DATE: np.datetime64 = np.datetime64(ISO_DATE)
TS_NP_FLOAT: np.float64 = np.float64(TS_PY_FLOAT)
TS_NP_INT: np.int64 = np.int64(TS_PY_INT)
TS_PA_DATE = pa.scalar(TS_PY_DATE, type=pa.timestamp("ms"))
TS_PD_DATE: pd.Timestamp = timestamp(ISO_DATE)
TIMESTAMPS: dict[str, TimeStamp] = {
    "float"       : TS_PY_FLOAT,
    "int"         : TS_PY_INT,
    "numpy"       : TS_NP_DATE,
    "numpy_float" : TS_NP_FLOAT,
    "numpy_int"   : TS_NP_INT,
    "pandas"      : TS_PD_DATE,
    "python"      : TS_PY_DATE,
    # NOT SUPPORTED: "arrow"       : TS_ARROW,
}  # fmt: skip
# endregion scalar datetimes -----------------------------------------------------------

# region scalar timedeltas -------------------------------------------------------------
TD_NP_FLOAT: np.float64 = np.float64(10.0)
TD_NP_INT: np.int64 = np.int64(10)
TD_NP_DUR: np.timedelta64 = np.timedelta64(1, "D")
TD_PA_DUR = pa.scalar(10, type=pa.duration("ms"))
TD_PD_DUR: pd.Timedelta = timedelta(days=1)
TD_PY_FLOAT: float = 10.0
TD_PY_INT: int = 10
TD_PY_DUR: python_timedelta = python_timedelta(days=1)

TIMEDELTAS: dict[str, TimeDelta] = {
    "numpy[float]"      : TD_NP_FLOAT,
    "numpy[int]"        : TD_NP_INT,
    "numpy[timedelta]"  : TD_NP_DUR,
    "pandas[timedelta]" : TD_PD_DUR,
    "python[float]"     : TD_PY_FLOAT,
    "python[int]"       : TD_PY_INT,
    "python[timedelta]" : TD_PY_DUR,
    # NOT SUPPORTED: "arrow"       : TD_ARROW,
}  # fmt: skip
# endregion scalar timedeltas ----------------------------------------------------------

# region array datetimes ---------------------------------------------------------------
TS_NDARRAY: NDArray[np.datetime64] = np.array([TS_NP_DATE])
TS_NDARRAY_FLOAT: NDArray[np.float64] = np.array([TS_NP_FLOAT])
TS_NDARRAY_INT: NDArray[np.int64] = np.array([TS_NP_INT])
# endregion array datetimes ------------------------------------------------------------

# region array timedeltas --------------------------------------------------------------
TD_NDARRAY_DUR: NDArray[np.timedelta64] = np.array([TD_PY_DUR], dtype="timedelta64[ns]")
TD_NDARRAY_FLOAT: NDArray[np.float64] = np.array([TD_PY_FLOAT], dtype="float64")
TD_NDARRAY_INT: NDArray[np.int64] = np.array([TD_PY_INT], dtype="int64")
PD_SERIES_NP_INT: pd.Series = pd.Series([TD_PY_INT], dtype="int64")
PD_SERIES_NP_FLOAT: pd.Series = pd.Series([TD_PY_FLOAT], dtype="float64")
PD_SERIES_NP_DUR: pd.Series = pd.Series([TD_PY_DUR], dtype="timedelta64[ns]")
PD_SERIES_PA_DUR: pd.Series = pd.Series([TD_PY_DUR], dtype="duration[ns][pyarrow]")
PD_SERIES_PA_INT: pd.Series = pd.Series([TD_PY_INT], dtype="int64[pyarrow]")
PD_SERIES_PA_FLOAT: pd.Series = pd.Series([TD_PY_FLOAT], dtype="float64[pyarrow]")
# endregion array timedeltas -----------------------------------------------------------

TD_ARRAYS = {
    "numpy[float]"      : TD_NDARRAY_FLOAT,
    "numpy[int]"        : TD_NDARRAY_INT,
    "numpy[timedelta]"  : TD_NDARRAY_DUR,
    "pandas[duration]"  : PD_SERIES_PA_DUR,
    "pandas[float]"     : PD_SERIES_NP_FLOAT,
    "pandas[int]"       : PD_SERIES_NP_INT,
    "pandas[timedelta]" : PD_SERIES_NP_DUR,
}  # fmt: skip

TD_PY_SCALARS = {
    "numpy[float]"      : TD_PY_FLOAT,
    "numpy[int]"        : TD_PY_INT,
    "numpy[timedelta]"  : TD_PY_DUR,
    "pandas[duration]"  : TD_PY_DUR,
    "pandas[float]"     : TD_PY_FLOAT,
    "pandas[int]"       : TD_PY_INT,
    "pandas[timedelta]" : TD_PY_DUR,
}  # fmt: skip

TS_PY_SCALARS = {
    "numpy[float]"      : TS_PY_FLOAT,
    "numpy[int]"        : TS_PY_INT,
    "numpy[timedelta]"  : TS_PY_DATE,
    "pandas[duration]"  : TS_PY_DATE,
    "pandas[float]"     : TS_PY_FLOAT,
    "pandas[int]"       : TS_PY_INT,
    "pandas[timedelta]" : TS_PY_DATE,
}  # fmt: skip


DATETIMES = {
    "numpy"      : TS_NP_DATE,
    "pandas"     : TS_PD_DATE,
    "python"     : TS_PY_DATE,
}  # fmt: skip
r"""All datetime objects."""


def test_timestamp_protocol_itself() -> None:
    r"""Test the datetime protocol."""
    non_callable_members = {
        member
        for member in get_protocol_members(TimeStamp)
        if not callable(getattr(TimeStamp, member))
    }
    assert not non_callable_members
    assert issubclass(TimeStamp, TimeStamp)


def test_timedelta_protocol_itself() -> None:
    r"""Test the datetime protocol."""
    non_callable_members = {
        member
        for member in get_protocol_members(TimeDelta)
        if not callable(getattr(TimeDelta, member))
    }
    assert not non_callable_members
    assert issubclass(TimeDelta, TimeDelta)


@pytest.mark.parametrize("name", TIMESTAMPS)
def test_timestamp_protocol(name: str) -> None:
    r"""Test the datetime protocol."""
    TS_value = TIMESTAMPS[name]
    assert isinstance(TS_value, TimeStamp)
    assert issubclass(type(TS_value), TimeStamp)

    # test __sub__
    zero = TS_value - TS_value
    assert isinstance(zero, TimeDelta)
    assert issubclass(type(zero), TimeDelta)

    # test __add__
    TS_new = TS_value + zero
    assert isinstance(TS_new, TimeStamp)
    assert issubclass(type(TS_new), TimeStamp)

    # test __ge__
    result = TS_value >= TS_value
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


def test_timestamp_assign() -> None:
    TS_float: TimeStamp[float] = TS_PY_FLOAT
    TS_int: TimeStamp[int] = TS_PY_INT
    TS_numpy: TimeStamp[np.timedelta64] = TS_NP_DATE
    TS_numpy_float: TimeStamp[np.float64] = TS_NP_FLOAT
    TS_numpy_int: TimeStamp[np.int64] = TS_NP_INT
    TS_pandas: TimeStamp[pd.Timedelta] = TS_PD_DATE
    TS_python: TimeStamp[python_timedelta] = TS_PY_DATE

    assert isinstance(TS_float, TimeStamp)
    assert isinstance(TS_int, TimeStamp)
    assert isinstance(TS_numpy, TimeStamp)
    assert isinstance(TS_numpy_float, TimeStamp)
    assert isinstance(TS_numpy_int, TimeStamp)
    assert isinstance(TS_pandas, TimeStamp)
    assert isinstance(TS_python, TimeStamp)


def test_timedelta_assign() -> None:
    td_float: TimeDelta = TD_PY_FLOAT
    td_int: TimeDelta = TD_PY_INT
    td_numpy: TimeDelta = TD_NP_DUR
    td_numpy_float: TimeDelta = TD_NP_FLOAT
    td_numpy_int: TimeDelta = TD_NP_INT
    td_pandas: TimeDelta = TD_PD_DUR
    td_python: TimeDelta = TD_PY_DUR

    assert isinstance(td_float, TimeDelta)
    assert isinstance(td_int, TimeDelta)
    assert isinstance(td_numpy, TimeDelta)
    assert isinstance(td_numpy_float, TimeDelta)
    assert isinstance(td_numpy_int, TimeDelta)
    assert isinstance(td_pandas, TimeDelta)
    assert isinstance(td_python, TimeDelta)


def test_timestamp_typevar() -> None:
    r"""Type-Checking TS_VAR."""

    def id_dt[DT: TimeStamp](x: DT, /) -> DT:
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

    def diff[TD: TimeDelta](x: TimeStamp[TD]) -> TD:
        return x - x

    assert_type(diff(TS_PY_FLOAT), float)
    assert_type(diff(TS_PY_INT), int)
    assert_type(diff(TS_NP_DATE), np.timedelta64)
    assert_type(diff(TS_NP_FLOAT), np.float64)  # type: ignore[assert-type, misc]
    assert_type(diff(TS_NP_INT), np.float32)  # type: ignore[assert-type, misc]
    assert_type(diff(TS_PD_DATE), pd.Timedelta)
    assert_type(diff(TS_PY_DATE), python_timedelta)


@pytest.mark.parametrize("example", TD_ARRAYS)
def test_timedelta_arrays(example: str) -> None:
    array = TD_ARRAYS[example]
    td_scalar = TD_PY_SCALARS[example]
    ts_scalar = TS_PY_SCALARS[example]
    cls = type(array)

    assert type(array + ts_scalar) is cls
    assert type(array + td_scalar) is cls
    assert type(array - td_scalar) is cls
    assert type(array < td_scalar) is cls


def test_td_var() -> None:
    r"""Type-Checking TD_VAR."""

    def id_td[TD: TimeDelta](x: TD, /) -> TD:
        return x

    id_td(TD_PY_FLOAT)
    id_td(TD_PY_INT)
    id_td(TD_NP_DUR)
    id_td(TD_NP_FLOAT)
    id_td(TD_NP_INT)
    id_td(TD_PD_DUR)
    id_td(TD_PY_DUR)


def test_joint_attrs_datetime() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(DATETIMES.values(), TimeStamp, raise_on_extra=False)


def test_joint_attrs_timestamp() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(TIMESTAMPS.values(), TimeStamp, raise_on_extra=False)


def test_joint_attrs_timedelta() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(TIMEDELTAS.values(), TimeDelta, raise_on_extra=False)
