r"""Tests for time related types."""

from datetime import datetime as py_datetime, timedelta as py_timedelta
from typing import Literal, assert_type

import numpy as np
import pandas as pd

from tsdm.testing import check_shared_interface, supports_issubclass
from tsdm.types.scalars import (
    DurationScalar,
    TimestampScalar,
)
from tsdm.utils import timedelta, timestamp

# region setup -------------------------------------------------------------------------
type np_int = np.int64  # noqa: PYI042
type np_float = np.float64  # noqa: PYI042
type np_timedelta = np.timedelta64[py_timedelta]  # noqa: PYI042
type np_datetime = np.datetime64[py_datetime]  # noqa: PYI042
type pd_datetime = pd.Timestamp  # noqa: PYI042
type pd_timedelta = pd.Timedelta  # noqa: PYI042

# fmt: off
ISO_DATE = "2021-01-01"
PY_FLOAT     : float        = float(10)
PY_INT       : int          = int(10.0)
PY_DATETIME  : py_datetime  = py_datetime.fromisoformat(ISO_DATE)
PY_TIMEDELTA : py_timedelta = py_timedelta(days=1)
NP_FLOAT     : np_float     = np.float64(10.0)
NP_INT       : np_int       = np.int64(10)
NP_TIMEDELTA : np_timedelta = np.timedelta64(1, "D")
NP_DATETIME  : np_datetime  = np.datetime64(ISO_DATE)
PD_DATETIME  : pd_datetime  = timestamp(ISO_DATE)
PD_TIMEDELTA : pd_timedelta = timedelta(days=1)
# fmt: on

type PY_TD = Literal["python[float]", "python[int]", "python[timedelta]"]
type PY_TS = Literal["python[float]", "python[int]", "python[datetime]"]
type NP_TD = Literal["numpy[float]", "numpy[int]", "numpy[timedelta]"]
type NP_TS = Literal["numpy[float]", "numpy[int]", "numpy[datetime]"]
type PD_TD = Literal["pandas[timedelta]"]
type PD_TS = Literal["pandas[datetime]"]
type DurationKey = PY_TD | NP_TD | PD_TD
type TimestampKey = PY_TS | NP_TS | PD_TS
# endregion setup ----------------------------------------------------------------------

# region test data ---------------------------------------------------------------------
TIMEDELTAS: dict[DurationKey, DurationScalar] = {
    "numpy[float]"      : NP_FLOAT,
    "numpy[int]"        : NP_INT,
    "numpy[timedelta]"  : NP_TIMEDELTA,  # type: ignore[dict-item]  # pyright: ignore[reportAssignmentType]
    "pandas[timedelta]" : PD_TIMEDELTA,
    "python[float]"     : PY_FLOAT,
    "python[int]"       : PY_INT,
    "python[timedelta]" : PY_TIMEDELTA,
}  # fmt: skip
r"""Dictionary of timedelta scalars."""

DURATION_TIMEDELTAS: dict[DurationKey, DurationScalar] = {
    "numpy[timedelta]"  : NP_TIMEDELTA,  # type: ignore[dict-item]  # pyright: ignore[reportAssignmentType]
    "pandas[timedelta]" : PD_TIMEDELTA,
    "python[timedelta]" : PY_TIMEDELTA,
}  # fmt: skip
r"""Dictionary of timedelta-like durations."""

TIMESTAMPS: dict[TimestampKey, TimestampScalar] = {
    "numpy[datetime]"  : NP_DATETIME,  # type: ignore[dict-item]  # pyright: ignore[reportAssignmentType]
    "numpy[float]"     : NP_FLOAT,
    "numpy[int]"       : NP_INT,
    "pandas[datetime]" : PD_DATETIME,
    "python[datetime]" : PY_DATETIME,
    "python[float]"    : PY_FLOAT,
    "python[int]"      : PY_INT,
}  # fmt: skip
r"""Dictionary of timestamp scalars."""

DATE_TIMESTAMPS: dict[TimestampKey, TimestampScalar[py_timedelta]] = {
    "numpy[datetime]"  : NP_DATETIME,   # type: ignore[dict-item]  # pyright: ignore[reportAssignmentType]
    "pandas[datetime]" : PD_DATETIME,
    "python[datetime]" : PY_DATETIME,
}  # fmt: skip
r"""Dictionary of datetime-like timestamps."""

FLOAT_TIMESTAMPS: dict[TimestampKey, TimestampScalar[float]] = {
    "numpy[float]"  : NP_FLOAT,
    "python[float]" : PY_FLOAT,
}  # fmt: skip
r"""Dictionary of float-like timestamps."""

INT_TIMESTAMPS: dict[TimestampKey, TimestampScalar[int]] = {
    "numpy[int]"  : NP_INT,  # type: ignore[dict-item]
    "python[int]" : PY_INT,
}  # fmt: skip
r"""Dictionary of int-like timestamps."""
# endregion test data ------------------------------------------------------------------


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


def test_timestamp_assign() -> None:
    assert isinstance(PY_FLOAT, TimestampScalar)
    assert isinstance(PY_INT, TimestampScalar)
    assert isinstance(NP_INT, TimestampScalar)
    assert isinstance(NP_FLOAT, TimestampScalar)
    assert isinstance(NP_DATETIME, TimestampScalar)
    assert isinstance(PD_DATETIME, TimestampScalar)
    assert isinstance(PY_DATETIME, TimestampScalar)


def test_timedelta_assign() -> None:
    assert isinstance(PY_FLOAT, DurationScalar)
    assert isinstance(PY_INT, DurationScalar)
    assert isinstance(NP_TIMEDELTA, DurationScalar)  # type: ignore[unreachable]
    assert isinstance(NP_FLOAT, DurationScalar)  # type: ignore[unreachable]
    assert isinstance(NP_INT, DurationScalar)
    assert isinstance(PD_TIMEDELTA, DurationScalar)
    assert isinstance(PY_TIMEDELTA, DurationScalar)


def type_assign_duration() -> None:
    r"""Test the datetime protocol."""
    # fmt: off
    _0: DurationScalar = NP_FLOAT
    _1: DurationScalar = NP_INT
    _2: DurationScalar = NP_TIMEDELTA  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _3: DurationScalar = PD_TIMEDELTA
    _4: DurationScalar = PY_FLOAT
    _5: DurationScalar = PY_INT
    _6: DurationScalar = PY_TIMEDELTA
    # fmt: on


def type_assign_timestamp_generic() -> None:
    r"""Test the datetime protocol."""
    # fmt: off
    _0: TimestampScalar = NP_DATETIME   # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _1: TimestampScalar = NP_INT
    _2: TimestampScalar = NP_FLOAT
    _3: TimestampScalar = PY_DATETIME
    _4: TimestampScalar = PY_INT
    _5: TimestampScalar = PY_FLOAT
    _6: TimestampScalar = PD_DATETIME
    # fmt: on


def type_assign_timestamp_basic() -> None:
    # fmt: off
    _0: TimestampScalar[float]        = PY_FLOAT
    _1: TimestampScalar[int]          = PY_INT
    _2: TimestampScalar[py_timedelta] = NP_DATETIME  # type:ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _3: TimestampScalar[float]        = NP_FLOAT
    _4: TimestampScalar[int]          = NP_INT       # type: ignore[assignment]
    _5: TimestampScalar[py_timedelta] = PD_DATETIME
    _6: TimestampScalar[py_timedelta] = PY_DATETIME
    # fmt: on


def type_assign_timestamp_concrete() -> None:
    # fmt: off
    _0: TimestampScalar[float]        = PY_FLOAT
    _1: TimestampScalar[int]          = PY_INT
    _2: TimestampScalar[np_timedelta] = NP_DATETIME  # type:ignore[type-var, assignment]  # pyright: ignore[reportAssignmentType, reportInvalidTypeArguments]
    _3: TimestampScalar[np_float]     = NP_FLOAT
    _4: TimestampScalar[np_int]       = NP_INT
    _5: TimestampScalar[pd_timedelta] = PD_DATETIME
    _6: TimestampScalar[py_timedelta] = PY_DATETIME
    # fmt: on


def type_duration_inference() -> None:
    r"""Check that DurationScalar can be inferred correctly."""

    def _id[TD: DurationScalar](x: TD, /) -> TD:
        return x

    # fmt: off
    assert_type( _id(PY_FLOAT)     , float        )
    assert_type( _id(PY_INT)       , int          )
    assert_type( _id(NP_FLOAT)     , np_float     )
    assert_type( _id(NP_INT)       , np_int       )
    assert_type( _id(NP_TIMEDELTA) , np_timedelta )  # type: ignore[type-var]  # pyright: ignore[reportAssertTypeFailure,reportArgumentType]
    assert_type( _id(PY_TIMEDELTA) , py_timedelta )
    assert_type( _id(PD_TIMEDELTA) , pd_timedelta )
    # fmt: on


def type_timestamp_inference() -> None:
    r"""Check that TimestampScalar can be inferred correctly."""

    def _id[DT: TimestampScalar](x: DT, /) -> DT:
        return x

    # fmt: off
    assert_type( _id(PY_FLOAT)    , float       )
    assert_type( _id(PY_INT)      , int         )
    assert_type( _id(NP_FLOAT)    , np_float    )
    assert_type( _id(NP_INT)      , np_int      )
    assert_type( _id(NP_DATETIME) , np_datetime )  # type: ignore[type-var]  # pyright: ignore[reportAssertTypeFailure,reportArgumentType]
    assert_type( _id(PY_DATETIME) , py_datetime )
    assert_type( _id(PD_DATETIME) , pd_datetime )
    # fmt: on


def type_timestamp_difference() -> None:
    r"""Check that TimestampScalar can be subtracted correctly."""
    # fmt: off
    assert_type( PY_FLOAT    - PY_FLOAT    , float        )
    assert_type( PY_INT      - PY_INT      , int          )
    assert_type( NP_FLOAT    - NP_FLOAT    , np_float     )
    assert_type( NP_INT      - NP_INT      , np_int       )
    assert_type( PY_DATETIME - PY_DATETIME , py_timedelta )
    assert_type( NP_DATETIME - NP_DATETIME , np_timedelta )
    assert_type( PD_DATETIME - PD_DATETIME , pd_timedelta )
    # fmt: on


def type_timestamp_difference_inference() -> None:
    r"""Check that differences between TimestampScalar can be inferred correctly."""

    def _sub[TD: DurationScalar](x: TimestampScalar[TD]) -> TD:
        return x - x

    # fmt: off
    assert_type( _sub(PY_FLOAT)    , float        )
    assert_type( _sub(PY_INT)      , int          )
    assert_type( _sub(NP_FLOAT)    , np_float     )
    assert_type( _sub(NP_INT)      , np_int       )  # type: ignore[assert-type, misc]  # pyright: ignore[reportAssertTypeFailure]
    assert_type( _sub(NP_DATETIME) , np_timedelta )  # type: ignore[assert-type, arg-type]  # pyright: ignore[reportAssertTypeFailure,reportArgumentType]
    assert_type( _sub(PY_DATETIME) , py_timedelta )  # type: ignore[assert-type, misc]
    assert_type( _sub(PD_DATETIME) , pd_timedelta )
    # fmt: on
