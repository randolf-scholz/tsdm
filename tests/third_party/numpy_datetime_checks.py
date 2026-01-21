r"""Check numpy datetime64 and timedelta64 types type inference."""
# mypy: enable-error-code="unused-ignore"

import datetime as dt
from typing import Protocol, Self, assert_type, overload

import numpy as np

type np_datetime = np.datetime64[dt.datetime]  # noqa: PYI042
type np_timedelta = np.timedelta64[dt.timedelta]  # noqa: PYI042

py_date = dt.date(year=2025, month=1, day=31)
py_dt = dt.datetime(year=2025, month=1, day=31, hour=1, minute=23, second=45)
py_td = dt.timedelta(seconds=37)

np_dt = np.datetime64(py_dt)
np_dt_date = np.datetime64(py_date)
np_dt_int = np.datetime64(1, "ps")
np_dt_nat = np.datetime64(None)

np_td = np.timedelta64(py_td)
np_td_int = np.timedelta64(1, "ps")
np_td_nat = np.timedelta64(None)


def check_static_types() -> None:
    r"""Check static types of various datetime and timedelta types."""
    # fmt: off
    # py_types
    assert_type(py_date, dt.date)
    assert_type(py_dt, dt.datetime)
    assert_type(py_td, dt.timedelta)
    # np_datetime64
    assert_type(np_dt,      "np.datetime64[dt.datetime]")
    assert_type(np_dt_date, "np.datetime64[dt.date]")
    assert_type(np_dt_int,  "np.datetime64[int]")
    assert_type(np_dt_nat,  "np.datetime64[None]")
    # np_timedelta64
    assert_type(np_td,      "np.timedelta64[dt.timedelta]")
    assert_type(np_td_int,  "np.timedelta64[int]")
    assert_type(np_td_nat,  "np.timedelta64[None]")
    # fmt: on


# ----------- runtime checks -----------------------------------------------------------
def test_runtime_subtraction_types() -> None:
    r"""Test runtime types of subtraction between various datetime and timedelta types."""
    # fmt: off
    # py_date
    assert type(py_date - py_td) is dt.date
    assert type(py_date - py_date) is dt.timedelta
    # py_dt
    assert type(py_dt - py_td) is dt.datetime
    assert type(py_dt - py_dt) is dt.timedelta
    # np_date
    assert type(np_dt_date - py_date)    is dt.timedelta
    # assert type(np_dt_date - py_dt)      is dt.timedelta
    assert type(np_dt_date - py_td)      is dt.date  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    assert type(np_dt_date - np_dt)      is np.timedelta64
    assert type(np_dt_date - np_dt_date) is np.timedelta64
    assert type(np_dt_date - np_dt_int)  is np.timedelta64
    assert type(np_dt_date - np_dt_nat)  is np.timedelta64
    assert type(np_dt_date - np_td)      is np.datetime64
    assert type(np_dt_date - np_td_int)  is np.datetime64
    assert type(np_dt_date - np_td_nat)  is np.datetime64
    # np_dt
    # assert type(np_dt - py_date)    is dt.timedelta
    assert type(np_dt - py_dt)      is dt.timedelta
    assert type(np_dt - py_td)      is dt.datetime  # type: ignore[operator]  # pyright: ignore[reportOperatorIssue]
    assert type(np_dt - np_dt)      is np.timedelta64
    assert type(np_dt - np_dt_date) is np.timedelta64
    assert type(np_dt - np_dt_int)  is np.timedelta64
    assert type(np_dt - np_dt_nat)  is np.timedelta64
    assert type(np_dt - np_td)      is np.datetime64
    assert type(np_dt - np_td_int)  is np.datetime64
    assert type(np_dt - np_td_nat)  is np.datetime64
    # np_dt_int
    # assert type(np_dt_int - py_date)    is dt.timedelta
    # assert type(np_dt_int - py_dt)      is dt.timedelta
    # assert type(np_dt_int - py_td)      is dt.datetime
    assert type(np_dt_int - np_dt)      is np.timedelta64
    assert type(np_dt_int - np_dt_date) is np.timedelta64
    assert type(np_dt_int - np_dt_int)  is np.timedelta64
    assert type(np_dt_int - np_dt_nat)  is np.timedelta64
    assert type(np_dt_int - np_td)      is np.datetime64
    assert type(np_dt_int - np_td_int)  is np.datetime64
    assert type(np_dt_int - np_td_nat)  is np.datetime64
    # np_nat
    # assert type(np_dt_nat - py_date)    is dt.timedelta
    # assert type(np_dt_nat - py_dt)      is dt.timedelta
    # assert type(np_dt_nat - py_td)      is dt.datetime
    assert type(np_dt_nat - np_dt)      is np.timedelta64
    assert type(np_dt_nat - np_dt_date) is np.timedelta64
    assert type(np_dt_nat - np_dt_int)  is np.timedelta64
    assert type(np_dt_nat - np_dt_nat)  is np.timedelta64
    assert type(np_dt_nat - np_td)      is np.datetime64
    assert type(np_dt_nat - np_td_int)  is np.datetime64
    assert type(np_dt_nat - np_td_nat)  is np.datetime64


# ---------- static checks -------------------------------------------------------------


def check_subtraction_static_types() -> None:
    r"""Check static types of subtraction between various datetime and timedelta types."""
    # fmt: off
    # py_date
    assert_type(py_date - py_td, dt.date)
    assert_type(py_date - py_date, dt.timedelta)
    # py_dt
    assert_type(py_dt - py_td, dt.datetime)
    assert_type(py_dt - py_dt, dt.timedelta)
    # np_dt
    # assert_type(np_dt - py_date,    dt.timedelta)
    assert_type(np_dt - py_dt,      dt.timedelta)
    assert_type(np_dt - py_td,      dt.datetime)  # type: ignore[assert-type, operator]  # pyright: ignore[reportOperatorIssue]
    assert_type(np_dt - np_dt,      "np.timedelta64[dt.timedelta]")
    assert_type(np_dt - np_dt_date, "np.timedelta64[dt.timedelta]")
    assert_type(np_dt - np_dt_int,  "np.timedelta64[int]")
    assert_type(np_dt - np_dt_nat,  "np.timedelta64[None]")
    assert_type(np_dt - np_td,      "np.datetime64[dt.datetime]")
    assert_type(np_dt - np_td_int,  "np.datetime64[int]")
    assert_type(np_dt - np_td_nat,  "np.datetime64[None]")
    # np_date
    assert_type(np_dt_date - py_date, "dt.timedelta")
    # assert_type(np_dt_date - py_dt,      "dt.timedelta")
    assert_type(np_dt_date - py_td,      "dt.date")  # type: ignore[assert-type, operator]  # pyright: ignore[reportOperatorIssue]
    assert_type(np_dt_date - np_dt,      "np.timedelta64[dt.timedelta]")
    assert_type(np_dt_date - np_dt_date, "np.timedelta64[dt.timedelta]")
    assert_type(np_dt_date - np_dt_int,  "np.timedelta64[int]")  # type: ignore[assert-type]
    assert_type(np_dt_date - np_dt_nat,  "np.timedelta64[None]")
    assert_type(np_dt_date - np_td,      "np.datetime64[dt.date]")
    assert_type(np_dt_date - np_td_int,  "np.datetime64[int]")  # type: ignore[assert-type]
    assert_type(np_dt_date - np_td_nat,  "np.datetime64[None]")
    # np_dt_int
    # assert_type(np_dt_int - py_date,    "dt.timedelta")
    # assert_type(np_dt_int - py_dt,      "dt.timedelta")
    # assert_type(np_dt_int - py_td,      "dt.date")  # ❌ raises [operator]
    assert_type(np_dt_int - np_dt,      "np.timedelta64[int]")
    assert_type(np_dt_int - np_dt_date, "np.timedelta64[int]")
    assert_type(np_dt_int - np_dt_int,  "np.timedelta64[int]")
    assert_type(np_dt_int - np_dt_nat,  "np.timedelta64[None]")  # type: ignore[assert-type]
    assert_type(np_dt_int - np_td,      "np.datetime64[int]")
    assert_type(np_dt_int - np_td_int,  "np.datetime64[int]")
    assert_type(np_dt_int - np_td_nat,  "np.datetime64[None]")  # type: ignore[assert-type]
    # np_nat
    assert_type(np_dt_nat - np_dt,      "np.timedelta64[None]")
    assert_type(np_dt_nat - np_dt_date, "np.timedelta64[None]")
    assert_type(np_dt_nat - np_dt_int,  "np.timedelta64[None]")
    assert_type(np_dt_nat - np_dt_nat,  "np.timedelta64[None]")
    assert_type(np_dt_nat - np_td,      "np.datetime64[None]")
    assert_type(np_dt_nat - np_td_int,  "np.datetime64[None]")
    assert_type(np_dt_nat - np_td_nat,  "np.datetime64[None]")
    # fmt: on


class Timedelta(Protocol):
    r"""Protocol for timedelta types."""

    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    def __sub__(self, other: Self, /) -> Self: ...
    def __rsub__(self, other: Self, /) -> Self: ...


class Timestamp[TD: Timedelta](Protocol):
    r"""Protocol for timestamp types."""

    @overload
    def __sub__(self, other: Self, /) -> TD: ...
    @overload
    def __sub__(self, other: TD, /) -> Self: ...


class SupportsSubSelf[TD: Timedelta](Protocol):
    r"""Protocol for types that support subtraction with self."""

    def __sub__(self, other: Self, /) -> TD: ...


class SupportsSubTD[TD: Timedelta](Protocol):
    r"""Protocol for types that support subtraction with timedelta."""

    def __sub__(self, other: TD, /) -> Self: ...


def static_check_protocols() -> None:
    r"""Check static types against defined protocols."""
    _1: SupportsSubTD = py_dt  # ✅️
    _2: SupportsSubTD = np_dt  # ✅️

    _3: SupportsSubSelf = py_dt  # ✅️
    _4: SupportsSubSelf = np_dt  # ✅️

    # w/o generic
    _5: Timestamp = py_dt  # ✅️
    _6: Timestamp = np_dt  # type: ignore[assignment]
    # w/ generic
    _7: Timestamp[dt.timedelta] = py_dt  # ✅️
    _8: Timestamp[np_timedelta] = np_dt  # type: ignore[assignment]
    # w/ nested generic
    _9: Timestamp[np_timedelta] = np_dt  # type: ignore[assignment]
