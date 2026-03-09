r"""Tests for time related types."""

from typing import Literal

from tsdm.experimental.types import SpanLikeScalar, TimeLikeScalar
from tsdm.testing import check_shared_interface, supports_issubclass

from .fixtures import SCALARS

# region setup -------------------------------------------------------------------------
type PY_TD = Literal[
    "python[types0d.py.float]", "python[types0d.py.int]", "python[timedelta]"
]
type PY_TS = Literal[
    "python[types0d.py.float]", "python[types0d.py.int]", "python[datetime]"
]
type NP_TD = Literal[
    "numpy[types0d.py.float]", "numpy[types0d.py.int]", "numpy[timedelta]"
]
type NP_TS = Literal[
    "numpy[types0d.py.float]", "numpy[types0d.py.int]", "numpy[datetime]"
]
type PD_TD = Literal["pandas[timedelta]"]
type PD_TS = Literal["pandas[datetime]"]
type DurationKey = PY_TD | NP_TD | PD_TD
type TimestampKey = PY_TS | NP_TS | PD_TS
# endregion setup ----------------------------------------------------------------------

# region test data ---------------------------------------------------------------------
TIMEDELTAS: dict[DurationKey, SpanLikeScalar] = {
    "numpy[types0d.py.float]"  : SCALARS.NP.FLOAT,
    "numpy[types0d.py.int]"    : SCALARS.NP.INT,
    "numpy[timedelta]"         : SCALARS.NP.TIMEDELTA,
    "pandas[timedelta]"        : SCALARS.PD.TIMEDELTA,
    "python[types0d.py.float]" : SCALARS.PY.FLOAT,
    "python[types0d.py.int]"   : SCALARS.PY.INT,
    "python[timedelta]"        : SCALARS.PY.TIMEDELTA,
}  # fmt: skip
r"""Dictionary of timedelta scalars."""

DURATION_TIMEDELTAS: dict[DurationKey, SpanLikeScalar] = {
    "numpy[timedelta]"  : SCALARS.NP.TIMEDELTA,
    "pandas[timedelta]" : SCALARS.PD.TIMEDELTA,
    "python[timedelta]" : SCALARS.PY.TIMEDELTA,
}  # fmt: skip
r"""Dictionary of timedelta-like durations."""

TIMESTAMPS: dict[TimestampKey, TimeLikeScalar] = {
    "numpy[datetime]"          : SCALARS.NP.DATETIME,  # type: ignore[dict-item]
    "numpy[types0d.py.float]"  : SCALARS.NP.FLOAT,
    "numpy[types0d.py.int]"    : SCALARS.NP.INT,
    "pandas[datetime]"         : SCALARS.PD.DATETIME,
    "python[datetime]"         : SCALARS.PY.DATETIME,
    "python[types0d.py.float]" : SCALARS.PY.FLOAT,
    "python[types0d.py.int]"   : SCALARS.PY.INT,
}  # fmt: skip
r"""Dictionary of timestamp scalars."""

DATE_TIMESTAMPS: dict[TimestampKey, TimeLikeScalar] = {
    "numpy[datetime]"  : SCALARS.NP.DATETIME,
    "pandas[datetime]" : SCALARS.PD.DATETIME,
    "python[datetime]" : SCALARS.PY.DATETIME,
}  # fmt: skip
r"""Dictionary of datetime-like timestamps."""

FLOAT_TIMESTAMPS: dict[TimestampKey, TimeLikeScalar] = {
    "numpy[types0d.py.float]"  : SCALARS.NP.FLOAT,
    "python[types0d.py.float]" : SCALARS.PY.FLOAT,
}  # fmt: skip
r"""Dictionary of float-like timestamps."""

INT_TIMESTAMPS: dict[TimestampKey, TimeLikeScalar] = {
    "numpy[types0d.py.int]"  : SCALARS.NP.INT,
    "python[types0d.py.int]" : SCALARS.PY.INT,
}  # fmt: skip
r"""Dictionary of int-like timestamps."""
# endregion test data ------------------------------------------------------------------


def test_timestamp_issubclass() -> None:
    r"""Test the datetime protocol."""
    assert supports_issubclass(TimeLikeScalar)


def test_timedelta_issubclass() -> None:
    r"""Test the datetime protocol."""
    assert supports_issubclass(SpanLikeScalar)


def test_joint_attrs_datetime() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(
        DATE_TIMESTAMPS.values(), TimeLikeScalar, raise_on_extra=False
    )


def test_joint_attrs_timestamp() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(TIMESTAMPS.values(), TimeLikeScalar, raise_on_extra=False)


def test_joint_attrs_timedelta() -> None:
    r"""Test the joint attributes of datetime objects."""
    check_shared_interface(TIMEDELTAS.values(), SpanLikeScalar, raise_on_extra=False)


def test_timestamp_assign() -> None:
    assert isinstance(SCALARS.PY.FLOAT, TimeLikeScalar)
    assert isinstance(SCALARS.PY.INT, TimeLikeScalar)
    assert isinstance(SCALARS.NP.INT, TimeLikeScalar)
    assert isinstance(SCALARS.NP.FLOAT, TimeLikeScalar)
    assert isinstance(SCALARS.NP.DATETIME, TimeLikeScalar)  # type: ignore[unreachable]
    assert isinstance(SCALARS.PD.DATETIME, TimeLikeScalar)  # type: ignore[unreachable]
    assert isinstance(SCALARS.PY.DATETIME, TimeLikeScalar)


def test_timedelta_assign() -> None:
    assert isinstance(SCALARS.PY.FLOAT, SpanLikeScalar)
    assert isinstance(SCALARS.PY.INT, SpanLikeScalar)
    assert isinstance(SCALARS.NP.TIMEDELTA, SpanLikeScalar)
    assert isinstance(SCALARS.NP.FLOAT, SpanLikeScalar)
    assert isinstance(SCALARS.NP.INT, SpanLikeScalar)
    assert isinstance(SCALARS.PD.TIMEDELTA, SpanLikeScalar)
    assert isinstance(SCALARS.PY.TIMEDELTA, SpanLikeScalar)
