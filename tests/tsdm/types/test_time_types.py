r"""Tests for time related types."""

from datetime import datetime, timedelta

import numpy
import pandas
from pytest import mark
from typing_extensions import get_protocol_members

from tsdm.types.time import DateTime, TimeDelta

ISO_DATE = "2021-01-01"

TIMEDELTAS: dict[str, TimeDelta] = {
    "float": float(10.0),
    "int": int(10),
    "numpy": numpy.timedelta64(1, "D"),
    "numpy_float": numpy.float64(10.0),
    "numpy_int": numpy.int64(10),
    "pandas": pandas.Timedelta(days=1),
    "python": timedelta(days=1),
    # "arrow": pyarrow.scalar(TD, type=pyarrow_td_type),
}


DATETIMES: dict[str, DateTime] = {  # pyright: ignore[reportGeneralTypeIssues]
    "float": float(10.0),
    "int": int(10),
    "numpy": numpy.datetime64(ISO_DATE),
    "numpy_float": numpy.float64(10.0),
    "numpy_int": numpy.int64(10),
    "pandas": pandas.Timestamp(ISO_DATE),
    "python": datetime.fromisoformat(ISO_DATE),
    # "arrow": pyarrow.scalar(DT, type=pyarrow.timestamp("s")),
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
