r"""Tests for time related types."""

from datetime import datetime, timedelta

import numpy
import numpy as np
import pandas
from pytest import mark
from typing_extensions import get_protocol_members

from tsdm.types.time import DT, TD, DateTime, TimeDelta

ISO_DATE = "2021-01-01"


DT_FLOAT = float(10.0)
DT_INT = int(10)
DT_NUMPY = numpy.datetime64(ISO_DATE)
DT_NUMPY_FLOAT = numpy.float64(10.0)
DT_NUMPY_INT = numpy.int64(10)
DT_PANDAS = pandas.Timestamp(ISO_DATE)
DT_PYTHON = datetime.fromisoformat(ISO_DATE)
DATETIMES: dict[str, DateTime] = {
    "float": DT_FLOAT,
    "int": DT_INT,
    "numpy": DT_NUMPY,
    "numpy_float": DT_NUMPY_FLOAT,
    "numpy_int": DT_NUMPY_INT,
    "pandas": DT_PANDAS,
    "python": DT_PYTHON,
    # "arrow": pyarrow.scalar(DT, type=pyarrow.timestamp("s")),
}

TD_FLOAT = float(10.0)
TD_INT = int(10)
TD_NUMPY = numpy.timedelta64(1, "D")
TD_NUMPY_FLOAT = numpy.float64(10.0)
TD_NUMPY_INT = numpy.int64(10)
TD_PANDAS = pandas.Timedelta(days=1)
TD_PYTHON = timedelta(days=1)
TIMEDELTAS: dict[str, TimeDelta] = {
    "float": TD_FLOAT,
    "int": TD_INT,
    "numpy": TD_NUMPY,
    "numpy_float": TD_NUMPY_FLOAT,
    "numpy_int": TD_NUMPY_INT,
    "pandas": TD_PANDAS,
    "python": TD_PYTHON,
    # "arrow": pyarrow.scalar(TD, type=pyarrow_td_type),
}


def test_datetime_protocol_itself() -> None:
    """Test the datetime protocol."""
    non_callable_members = {
        member
        for member in get_protocol_members(DateTime)
        if not callable(getattr(DateTime, member))
    }
    assert not non_callable_members
    assert issubclass(DateTime, DateTime)


def test_timedelta_protocol_itself() -> None:
    """Test the datetime protocol."""
    non_callable_members = {
        member
        for member in get_protocol_members(TimeDelta)
        if not callable(getattr(TimeDelta, member))
    }
    assert not non_callable_members
    assert issubclass(TimeDelta, TimeDelta)


@mark.parametrize("name", DATETIMES)
def test_datetime_protocol(name: str) -> None:
    """Test the datetime protocol."""
    dt_value = DATETIMES[name]
    assert isinstance(dt_value, DateTime)
    assert issubclass(type(dt_value), DateTime)
    zero = dt_value - dt_value
    assert isinstance(zero, TimeDelta)
    assert issubclass(type(zero), TimeDelta)
    dt_new = dt_value + zero
    assert isinstance(dt_new, DateTime)
    assert issubclass(type(dt_new), DateTime)


@mark.parametrize("name", TIMEDELTAS)
def test_timedelta_protocol(name: str) -> None:
    """Test the datetime protocol."""
    td_value = TIMEDELTAS[name]
    assert isinstance(td_value, TimeDelta)
    assert issubclass(type(td_value), TimeDelta)
    zero = td_value - td_value
    assert isinstance(zero, TimeDelta)
    assert issubclass(type(zero), TimeDelta)


def test_datetime_joint_attrs() -> None:
    """Test the joint attributes of datetime objects."""
    shared_attrs = set.intersection(*(set(dir(v)) for v in DATETIMES.values()))
    superfluous_attrs = shared_attrs - set(dir(DateTime))
    print(f"\nShared members not covered by DateTime:\n\t{superfluous_attrs}")


def test_timedelta_joint_attrs() -> None:
    """Test the joint attributes of datetime objects."""
    shared_attrs = set.intersection(*(set(dir(v)) for v in TIMEDELTAS.values()))
    superfluous_attrs = shared_attrs - set(dir(TimeDelta))
    print(f"\nShared members not covered by TimeDelta:\n\t{superfluous_attrs}")


def test_datetime_assign() -> None:
    dt_float: DateTime[float] = DT_FLOAT
    dt_int: DateTime[int] = DT_INT
    dt_numpy: DateTime[np.timedelta64] = DT_NUMPY
    dt_numpy_float: DateTime[np.float64] = DT_NUMPY_FLOAT
    dt_numpy_int: DateTime[np.int64] = DT_NUMPY_INT
    dt_pandas: DateTime[pandas.Timedelta] = DT_PANDAS
    dt_python: DateTime[timedelta] = DT_PYTHON

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
    """Type-Checking DT_VAR."""
    # TODO: submit issue to pyright

    def id_dt(x: DT, /) -> DT:
        return x

    id_dt(DT_FLOAT)
    id_dt(DT_INT)
    id_dt(DT_NUMPY)
    id_dt(DT_NUMPY_FLOAT)
    id_dt(DT_NUMPY_INT)
    id_dt(DT_PANDAS)
    id_dt(DT_PYTHON)


def test_td_var() -> None:
    """Type-Checking TD_VAR."""

    def id_td(x: TD, /) -> TD:
        return x

    id_td(TD_FLOAT)
    id_td(TD_INT)
    id_td(TD_NUMPY)
    id_td(TD_NUMPY_FLOAT)
    id_td(TD_NUMPY_INT)
    id_td(TD_PANDAS)
    id_td(TD_PYTHON)
