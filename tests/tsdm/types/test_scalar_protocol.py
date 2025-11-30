r"""Tests for `tsdm.types.scalars`."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest
import torch as pt

from tsdm.testing import check_shared_interface
from tsdm.types.scalars import (
    AdditiveScalar,
    BoolScalar,
    ComplexScalar,
    DurationScalar,
    FloatScalar,
    IntScalar,
    OrderedScalar,
    TimestampScalar,
)

BOOL: bool = bool(1)
INT: int = int(1.0)
FLOAT: float = float(1)
COMPLEX: complex = complex(0 + 1j)
DATETIME: dt.datetime = dt.datetime(2021, 1, 1)
TIMEDELTA: dt.timedelta = dt.timedelta(days=1)

BASE_SCALARS: dict[object, object] = {
    None         : None,
    bool         : True,
    int          : 1,
    float        : 1.0,
    complex      : 1 + 1j,
    dt.datetime  : dt.datetime(2021, 1, 1),
    dt.timedelta : dt.timedelta(days=1),
    pd.NA        : pd.NA,
    pd.NaT       : pd.NaT,
}  # fmt: skip
r"""Base scalars for testing."""

ORDERED_SCALARS: dict[str, OrderedScalar] = {
    "np_bool"      : np.True_,
    "np_datetime"  : np.datetime64("2021-01-01"),
    "np_float"     : np.float64(1.0),
    "np_int"       : np.int64(1),
    "np_timedelta" : np.timedelta64(1, "D"),
    "pd_timedelta" : pd.Timedelta("1D"),
    "pd_timestamp" : pd.Timestamp("2021-01-01"),
    "py_string"    : "1",
    "py_bytes"     : b"1",
    "py_bool"      : True,
    "py_tuple"     : (1,),
    "py_datetime"  : dt.datetime(2021, 1, 1),
    "py_float"     : 1.0,
    "py_int"       : 1,
    "py_timedelta" : dt.timedelta(days=1),
    "pt_bool"      : pt.tensor([True], dtype=pt.bool),
    "pt_int"       : pt.tensor(1, dtype=pt.int64),
    "pt_float"     : pt.tensor(1.0, dtype=pt.float64),
}  # fmt: skip
r"""Ordered scalars for testing."""

ADDITIVE_SCALARS: dict[str, AdditiveScalar] = {
    "np_complex"   : np.complex128(1 + 1j),
    "np_float"     : np.float64(1.0),
    "np_int"       : np.int64(1),
    "np_timedelta" : np.timedelta64(1, "D"),
    "pd_timedelta" : pd.Timedelta("1D"),
    "pt_complex"   : pt.tensor(1 + 1j, dtype=pt.complex128),
    "pt_float"     : pt.tensor(1.0, dtype=pt.float64),
    "pt_int"       : pt.tensor(1, dtype=pt.int64),
    "py_complex"   : 1 + 1j,
    "py_float"     : 1.0,
    "py_int"       : 1,
    "py_timedelta" : dt.timedelta(days=1),
}  # fmt: skip
r"""Additive scalars for testing."""

BOOLEAN_SCALARS: dict[str, BoolScalar] = {
    "np_bool"    : np.bool(bool(1)),
    "np_literal" : np.True_,  # type: ignore[dict-item] # pyright: ignore[reportAssignmentType]
    "py_bool"    : bool(1),
    "py_literal" : True,
    "pt_bool"    : pt.tensor([True], dtype=pt.bool),
}  # fmt: skip
r"""Boolean scalars for testing."""

INT_SCALARS: dict[str, IntScalar] = {
    "np_int"     : np.int64(1),
    "py_int"     : int(1.0),  # type: ignore[dict-item]
    "py_literal" : 1,  # type: ignore[dict-item]
    "pt_int"     : pt.tensor(1, dtype=pt.int64),
}  # fmt: skip
r"""Integer scalars for testing."""

FLOAT_SCALARS: dict[str, FloatScalar] = {
    "np_float"   : np.float64(1.0),
    "py_float"   : float(1),
    "py_literal" : 1.0,
    "pt_float"   : pt.tensor(1.0, dtype=pt.float64),
}  # fmt: skip
r"""Float scalars for testing."""

COMPLEX_SCALARS: dict[str, ComplexScalar] = {
    "np_complex" : np.complex128(1 + 1j),
    "py_complex" : complex(1 + 1j),
    "py_literal" : 1 + 1j,
    "pt_complex" : pt.tensor(1 + 1j, dtype=pt.complex128),
}  # fmt: skip
r"""Complex scalars for testing."""

TIMEDELTA_SCALARS: dict[str, DurationScalar] = {
    "np_float" : np.float64(1.0),
    "np_int"   : np.int64(1),
    "np_time"  : np.timedelta64(1, "D"),  # type: ignore[dict-item] # pyright: ignore[reportAssignmentType]
    "pd_time"  : pd.Timedelta("1D"),
    "py_float" : 1.0,
    "py_int"   : 1,
    "py_time"  : dt.timedelta(days=1),
}  # fmt: skip
r"""Dictionary of timedelta scalars."""

TIMESTAMP_SCALARS: dict[str, TimestampScalar] = {
    "np_time"  : np.datetime64("2021-01-01"),  # type: ignore[dict-item] # pyright: ignore[reportAssignmentType]
    "np_float" : np.float64(1.0),
    "np_int"   : np.int64(1),
    "pd_time"  : pd.Timestamp("2021-01-01"),
    "py_time"  : dt.datetime(2021, 1, 1),
    "py_float" : float(1),
    "py_int"   : int(1.0),
}  # fmt: skip
r"""Dictionary of timestamp scalars."""

TEST_TYPED_CASES: dict[type, dict] = {
    BoolScalar      : BOOLEAN_SCALARS,
    ComplexScalar   : COMPLEX_SCALARS,
    FloatScalar     : FLOAT_SCALARS,
    IntScalar       : INT_SCALARS,
    DurationScalar  : TIMEDELTA_SCALARS,
    TimestampScalar : TIMESTAMP_SCALARS,
}  # fmt: skip
r"""Test cases for scalar types."""


@pytest.mark.parametrize("name", BOOLEAN_SCALARS)
def test_boolean_scalar(name: str) -> None:
    value = BOOLEAN_SCALARS[name]
    cls = type(value)
    assert isinstance(value, BoolScalar)

    # fmt: off
    # region test unary operations
    # test conversions
    assert type(bool(value)) is bool       # __bool__
    assert type(int(value)) is int         # __int__
    assert type(float(value)) is float     # __float_
    # endregion test unary operations
    # region test binary operations
    # test boolean operators (self)
    assert type(value & value) is cls      # __and__
    assert type(value ^ value) is cls      # __xor__
    assert type(value | value) is cls      # __or__
    # test boolean operators (bool)
    assert type(value & BOOL) is cls       # __and__
    assert type(value | BOOL) is cls       # __or__
    assert type(value ^ BOOL) is cls       # __xor__
    # test boolean operators (bool, reversed)
    assert type(BOOL & value) is cls       # __rand__
    assert type(BOOL | value) is cls       # __ror__
    assert type(BOOL ^ value) is cls       # __rxor__
    # endregion test binary operations
    # region test comparisons
    # test comparisons (self)
    assert type(value == value) is cls     # __eq__
    assert type(value != value) is cls     # __ne__
    assert type(value <  value) is cls     # __lt__
    assert type(value <= value) is cls     # __le__
    assert type(value >  value) is cls     # __gt__
    assert type(value >= value) is cls     # __ge__
    # test comparisons (bool)
    assert type(value != BOOL) is cls      # __ne__
    assert type(value <  BOOL) is cls      # __lt__
    assert type(value <= BOOL) is cls      # __le__
    assert type(value >  BOOL) is cls      # __gt__
    assert type(value >= BOOL) is cls      # __ge__
    # test comparisons (int)
    assert type(value == INT) is cls       # __eq__
    assert type(value != INT) is cls       # __ne__
    assert type(value <  INT) is cls       # __lt__
    assert type(value <= INT) is cls       # __le__
    assert type(value >  INT) is cls       # __gt__
    assert type(value >= INT) is cls       # __ge__
    # test comparisons (float)
    assert type(value == FLOAT) is cls     # __eq__
    assert type(value != FLOAT) is cls     # __ne__
    assert type(value <  FLOAT) is cls     # __lt__
    assert type(value <= FLOAT) is cls     # __le__
    assert type(value >  FLOAT) is cls     # __gt__
    assert type(value >= FLOAT) is cls     # __ge__
    # test comparisons (complex)
    assert type(value == COMPLEX) is cls   # __eq__
    assert type(value != COMPLEX) is cls   # __ne__
    # endregion test comparisons
    # fmt: on


@pytest.mark.parametrize("name", INT_SCALARS)
def test_int_scalar(name: str) -> None:
    value: IntScalar = INT_SCALARS[name]
    cls = type(value)
    assert isinstance(value, IntScalar)

    # fmt: off
    # region test unary operations
    # test conversions
    assert type(bool(value)) is bool         # __bool__
    assert type(int(value)) is int           # __int__
    assert type(value.__index__()) is int    # __index__
    assert type(float(value)) is float       # __float_
    # test unary operations
    assert type(abs(value)) is cls           # __abs__
    assert type(-value) is cls               # __neg__
    assert type(+value) is cls               # __pos__
    # endregion test unary operations
    # region test binary operations
    # test arithmetic operations (self)
    assert type(value +  value) is cls  # __add__
    assert type(value -  value) is cls  # __sub__
    assert type(value *  value) is cls  # __mul__
    assert type(value ** value) is cls  # __pow__
    assert type(value // value) is cls  # __floordiv__
    assert type(value %  value) is cls  # __mod__
    # test arithmetic operations (int)
    assert type(value +  INT) is cls    # __add__
    assert type(value -  INT) is cls    # __sub__
    assert type(value *  INT) is cls    # __mul__
    assert type(value ** INT) is cls    # __pow__
    assert type(value // INT) is cls    # __floordiv__
    assert type(value %  INT) is cls    # __mod__
    # test arithmetic operations (int, reversed)
    assert type(INT +  value) is cls    # __radd__
    assert type(INT -  value) is cls    # __rsub__
    assert type(INT *  value) is cls    # __rmul__
    assert type(INT ** value) is cls    # __rpow__
    assert type(INT // value) is cls    # __rfloordiv__
    assert type(INT %  value) is cls    # __rmod__
    # test arithmetic operations (bool)
    assert type(value +  BOOL) is cls   # __add__
    # assert type(value -  BOOL) is cls   # __sub__
    assert type(value *  BOOL) is cls   # __mul__
    assert type(value ** BOOL) is cls   # __pow__
    assert type(value // BOOL) is cls   # __floordiv__
    assert type(value %  BOOL) is cls   # __mod__
    # test arithmetic operations (bool, reversed)
    assert type(BOOL +  value) is cls   # __radd__
    # assert type(BOOL -  value) is cls   # __rsub__
    assert type(BOOL *  value) is cls   # __rmul__
    assert type(BOOL ** value) is cls   # __rpow__
    assert type(BOOL // value) is cls   # __rfloordiv__
    assert type(BOOL %  value) is cls   # __rmod__
    # endregion test binary operations
    # region test comparisons
    # test comparisons (self)
    assert isinstance(value == value, BoolScalar)     # __eq__
    assert isinstance(value != value, BoolScalar)     # __ne__
    assert isinstance(value <  value, BoolScalar)     # __lt__
    assert isinstance(value <= value, BoolScalar)     # __le__
    assert isinstance(value >  value, BoolScalar)     # __gt__
    assert isinstance(value >= value, BoolScalar)     # __ge__
    # test comparisons (bool)
    assert isinstance(value == BOOL, BoolScalar)      # __eq__
    assert isinstance(value != BOOL, BoolScalar)      # __ne__
    assert isinstance(value <  BOOL, BoolScalar)      # __lt__
    assert isinstance(value <= BOOL, BoolScalar)      # __le__
    assert isinstance(value >  BOOL, BoolScalar)      # __gt__
    assert isinstance(value >= BOOL, BoolScalar)      # __ge__
    # test comparisons (int)
    assert isinstance(value == INT, BoolScalar)       # __eq__
    assert isinstance(value != INT, BoolScalar)       # __ne__
    assert isinstance(value <  INT, BoolScalar)       # __lt__
    assert isinstance(value <= INT, BoolScalar)       # __le__
    assert isinstance(value >  INT, BoolScalar)       # __gt__
    assert isinstance(value >= INT, BoolScalar)       # __ge__
    # test comparisons (float)
    assert isinstance(value == FLOAT, BoolScalar)     # __eq__
    assert isinstance(value != FLOAT, BoolScalar)     # __ne__
    assert isinstance(value <  FLOAT, BoolScalar)     # __lt__
    assert isinstance(value <= FLOAT, BoolScalar)     # __le__
    assert isinstance(value >  FLOAT, BoolScalar)     # __gt__
    assert isinstance(value >= FLOAT, BoolScalar)     # __ge__
    # test comparisons (complex)
    assert isinstance(value == COMPLEX, BoolScalar)   # __eq__
    assert isinstance(value != COMPLEX, BoolScalar)   # __ne__
    # endregion test comparisons
    # fmt: on


@pytest.mark.parametrize("name", FLOAT_SCALARS)
def test_float_scalar(name: str) -> None:
    value: FloatScalar = FLOAT_SCALARS[name]
    cls = type(value)
    assert isinstance(value, FloatScalar)

    # fmt: off
    # region test unary operations
    # test conversions
    assert type(bool(value)) is bool       # __bool__
    assert type(int(value)) is int         # __int__
    assert type(float(value)) is float     # __float_
    # test unary operations
    assert type(abs(value)) is cls       # __abs__
    assert type(-value) is cls           # __neg__
    assert type(+value) is cls           # __pos__
    # endregion test unary operations
    # region test binary operations
    # test arithmetic operations (self)
    assert type(value + value) is cls    # __add__
    assert type(value - value) is cls    # __sub__
    assert type(value * value) is cls    # __mul__
    assert type(value / value) is cls    # __truediv__
    assert type(value**value) is cls     # __pow__
    assert type(value // value) is cls   # __floordiv__
    # test arithmetic operations (float)
    assert type(value +  FLOAT) is cls   # __add__
    assert type(value -  FLOAT) is cls   # __sub__
    assert type(value *  FLOAT) is cls   # __mul__
    assert type(value /  FLOAT) is cls   # __truediv__
    assert type(value ** FLOAT) is cls   # __pow__
    assert type(value // FLOAT) is cls   # __floordiv__
    # test arithmetic operations (float, reversed)
    assert type(FLOAT +  value) is cls   # __radd__
    assert type(FLOAT -  value) is cls   # __rsub__
    assert type(FLOAT *  value) is cls   # __rmul__
    assert type(FLOAT /  value) is cls   # __rtruediv__
    assert type(FLOAT ** value) is cls   # __rpow__
    assert type(FLOAT // value) is cls   # __rfloordiv__
    # endregion test binary operations
    # region test comparisons
    # test comparisons (self)
    assert isinstance(value == value, BoolScalar)     # __eq__
    assert isinstance(value != value, BoolScalar)     # __ne__
    assert isinstance(value <  value, BoolScalar)     # __lt__
    assert isinstance(value <= value, BoolScalar)     # __le__
    assert isinstance(value >  value, BoolScalar)     # __gt__
    assert isinstance(value >= value, BoolScalar)     # __ge__
    # test comparisons (bool)
    assert isinstance(value == BOOL, BoolScalar)      # __eq__
    assert isinstance(value != BOOL, BoolScalar)      # __ne__
    assert isinstance(value <  BOOL, BoolScalar)      # __lt__
    assert isinstance(value <= BOOL, BoolScalar)      # __le__
    assert isinstance(value >  BOOL, BoolScalar)      # __gt__
    assert isinstance(value >= BOOL, BoolScalar)      # __ge__
    # test comparisons (int)
    assert isinstance(value == INT, BoolScalar)       # __eq__
    assert isinstance(value != INT, BoolScalar)       # __ne__
    assert isinstance(value <  INT, BoolScalar)       # __lt__
    assert isinstance(value <= INT, BoolScalar)       # __le__
    assert isinstance(value >  INT, BoolScalar)       # __gt__
    assert isinstance(value >= INT, BoolScalar)       # __ge__
    # test comparisons (float)
    assert isinstance(value == FLOAT, BoolScalar)     # __eq__
    assert isinstance(value != FLOAT, BoolScalar)     # __ne__
    assert isinstance(value <  FLOAT, BoolScalar)     # __lt__
    assert isinstance(value <= FLOAT, BoolScalar)     # __le__
    assert isinstance(value >  FLOAT, BoolScalar)     # __gt__
    assert isinstance(value >= FLOAT, BoolScalar)     # __ge__
    # test comparisons (complex)
    assert isinstance(value == COMPLEX, BoolScalar)   # __eq__
    assert isinstance(value != COMPLEX, BoolScalar)   # __ne__
    # endregion test comparisons
    # fmt: on


@pytest.mark.parametrize("name", COMPLEX_SCALARS)
def test_complex_scalar(name: str) -> None:
    value = COMPLEX_SCALARS[name]
    cls = type(value)
    assert isinstance(value, ComplexScalar)

    # fmt: off
    # region test unary operations
    # test conversions
    assert type(bool(value)) is bool            # __bool__
    assert type(complex(value)) is complex      # __complex__
    # test unary operations
    assert isinstance(abs(value), FloatScalar)  # __abs__
    assert type(-value) is cls                  # __neg__
    assert type(+value) is cls                  # __pos__
    # endregion test unary operations
    # region test binary operations
    # test arithmetic operations (self)
    assert type(value +  value) is cls    # __add__
    assert type(value -  value) is cls    # __sub__
    assert type(value *  value) is cls    # __mul__
    assert type(value /  value) is cls    # __truediv__
    assert type(value ** value) is cls    # __pow__
    # test arithmetic operations (complex)
    assert type(value +  COMPLEX) is cls  # __add__
    assert type(value -  COMPLEX) is cls  # __sub__
    assert type(value *  COMPLEX) is cls  # __mul__
    assert type(value /  COMPLEX) is cls  # __truediv__
    assert type(value ** COMPLEX) is cls  # __pow__
    # test arithmetic operations (complex, reversed)
    assert type(COMPLEX +  value) is cls  # __radd__
    assert type(COMPLEX -  value) is cls  # __rsub__
    assert type(COMPLEX *  value) is cls  # __rmul__
    assert type(COMPLEX /  value) is cls  # __rtruediv__
    assert type(COMPLEX ** value) is cls  # __rpow__
    # test arithmetic operations (float)
    assert type(value +  FLOAT) is cls    # __add__
    assert type(value -  FLOAT) is cls    # __sub__
    assert type(value *  FLOAT) is cls    # __mul__
    assert type(value /  FLOAT) is cls    # __truediv__
    assert type(value ** FLOAT) is cls    # __pow__
    # test arithmetic operations (float, reversed)
    assert type(FLOAT +  value) is cls    # __radd__
    assert type(FLOAT -  value) is cls    # __rsub__
    assert type(FLOAT *  value) is cls    # __rmul__
    assert type(FLOAT /  value) is cls    # __rtruediv__
    assert type(FLOAT ** value) is cls    # __rpow__
    # endregion test binary operations
    # region test comparisons
    # test comparisons (self)
    assert isinstance(value == value, BoolScalar)     # __eq__
    assert isinstance(value != value, BoolScalar)     # __ne__
    # test comparisons (bool)
    assert isinstance(value == BOOL, BoolScalar)      # __eq__
    assert isinstance(value != BOOL, BoolScalar)      # __ne__
    # test comparisons (int)
    assert isinstance(value == INT, BoolScalar)       # __eq__
    assert isinstance(value != INT, BoolScalar)       # __ne__
    # test comparisons (float)
    assert isinstance(value == FLOAT, BoolScalar)     # __eq__
    assert isinstance(value != FLOAT, BoolScalar)     # __ne__
    # test comparisons (complex)
    assert isinstance(value == COMPLEX, BoolScalar)   # __eq__
    assert isinstance(value != COMPLEX, BoolScalar)   # __ne__
    # endregion test comparisons
    # fmt: on


@pytest.mark.parametrize("name", TIMEDELTA_SCALARS)
def test_timedelta_scalar(name: str) -> None:
    value = TIMEDELTA_SCALARS[name]
    cls = type(value)
    assert isinstance(value, DurationScalar)

    # fmt: off
    # region test unary operations
    assert type(abs(value)) is cls      # __abs__
    assert type(-value) is cls          # __neg__
    assert type(+value) is cls          # __pos__
    # endregion test unary operations
    # region test binary operations
    # test arithmetic operations (self)
    assert type(value + value) is cls               # __add__
    assert type(value - value) is cls               # __sub__
    assert type(value % value) is cls               # __mod__
    assert isinstance(value / value, FloatScalar)   # __truediv__
    assert isinstance(value // value, FloatScalar)  # __truediv__
    # test arithmetic operations (int)
    assert type(value *  INT) is cls                # __mul__
    assert type(INT * value) is cls                 # __rmul__
    # endregion test binary operations
    # region test comparisons
    # test comparisons (self)
    assert isinstance(value == value, BoolScalar)   # __eq__
    assert isinstance(value != value, BoolScalar)   # __ne__
    assert isinstance(value <  value, BoolScalar)   # __lt__
    assert isinstance(value <= value, BoolScalar)   # __le__
    assert isinstance(value >  value, BoolScalar)   # __gt__
    assert isinstance(value >= value, BoolScalar)   # __ge__
    # endregion test comparisons
    # fmt: on


@pytest.mark.parametrize("name", TIMESTAMP_SCALARS)
def test_timestamp_scalar(name: str) -> None:
    value = TIMESTAMP_SCALARS[name]
    cls = type(value)
    assert isinstance(value, TimestampScalar)
    ZERO = value - value
    # fmt: off
    # region test binary operations
    assert type(value + ZERO) is cls                  # __add__
    assert type(ZERO + value) is cls                  # __radd__
    assert isinstance(value - value, DurationScalar)  # __sub__
    # endregion test binary operations
    # region test comparisons
    assert isinstance(value >  value, BoolScalar)
    assert isinstance(value <  value, BoolScalar)
    assert isinstance(value >= value, BoolScalar)
    assert isinstance(value <= value, BoolScalar)
    assert isinstance(value == value, BoolScalar)
    assert isinstance(value != value, BoolScalar)
    # endregion test comparisons
    # fmt: on


@pytest.mark.parametrize("name", ORDERED_SCALARS)
def test_ordered_scalar(name: str) -> None:
    value = ORDERED_SCALARS[name]
    assert isinstance(value, OrderedScalar)

    # fmt: off
    # test comparisons (self)
    assert isinstance(value == value, BoolScalar)     # __eq__
    assert isinstance(value != value, BoolScalar)     # __ne__
    assert isinstance(value <  value, BoolScalar)     # __lt__
    assert isinstance(value <= value, BoolScalar)     # __le__
    assert isinstance(value >  value, BoolScalar)     # __gt__
    assert isinstance(value >= value, BoolScalar)     # __ge__
    # value check comparisons
    assert value == value
    assert value <= value
    assert value >= value
    assert not (value != value)  # noqa: SIM202
    assert not (value > value)
    assert not (value < value)
    # fmt: on


@pytest.mark.parametrize("name", ADDITIVE_SCALARS)
def test_additive_scalar(name: str) -> None:
    value = ADDITIVE_SCALARS[name]
    cls = type(value)
    assert isinstance(value, AdditiveScalar)

    assert type(value + value) is cls  # __add__
    assert type(value - value) is cls  # __sub__


@pytest.mark.parametrize("protocol", TEST_TYPED_CASES)
def test_shared_interface(protocol: type) -> None:
    test_cases = TEST_TYPED_CASES[protocol]
    check_shared_interface(test_cases.values(), protocol, raise_on_extra=False)


def type_float_scalar() -> None:
    _1: FloatScalar = np.floating()


def type_timestamp_assignable() -> None:
    # numpy
    _np_0: TimestampScalar[np.int64] = np.int64(0)
    _np_1: TimestampScalar[np.float64] = np.float64(1)
    # FIXME: https://github.com/numpy/numpy/issues/28257
    _np_2: TimestampScalar[np.timedelta64] = np.datetime64("2021-01-01")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _np_3: TimestampScalar[dt.timedelta] = np.datetime64("2021-01-01")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # python
    _py_1: TimestampScalar[dt.timedelta] = dt.datetime(2021, 1, 1)
    _py_2: TimestampScalar[int] = int(3)
    _py_3: TimestampScalar[float] = float(3.0)
    # pandas
    _pd_1: TimestampScalar[dt.timedelta] = pd.Timestamp("2021-01-01")
    _pd_2: TimestampScalar[pd.Timedelta] = pd.Timestamp("2021-01-01")


def type_duration_assignable() -> None:
    # numpy
    _np_0: DurationScalar = np.int64(0)
    _np_1: DurationScalar = np.float64(1)
    # FIXME: https://github.com/numpy/numpy/issues/28257
    _np_2: DurationScalar = np.timedelta64(1, "D")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _np_3: DurationScalar = np.timedelta64(1, "D")  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # python
    _py_1: DurationScalar = dt.timedelta(days=1)
    _py_2: DurationScalar = int(3)
    _py_3: DurationScalar = float(3.0)
    # pandas
    _pd_1: DurationScalar = pd.Timedelta(days=1)


def type_boolean_assignable() -> None:
    # python
    _py_0: BoolScalar = bool(1234)
    _py_2: BoolScalar = True
    _py_3: BoolScalar = False
    # numpy
    _np_0: BoolScalar = np.bool_(bool(1234))
    _np_1: BoolScalar = np.True_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    _np_2: BoolScalar = np.False_  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
    # pytorch
    _pt_0: BoolScalar = pt.tensor([True], dtype=pt.bool)


def type_int_assignable() -> None:
    # python
    _py_0: IntScalar = int(1234)  # type: ignore[assignment]
    _py_1: IntScalar = 0  # type: ignore[assignment]
    # numpy
    _np_0: IntScalar = np.int64(1234)
    # pytorch
    _pt_0: IntScalar = pt.tensor([1234], dtype=pt.int64)


def type_float_assignable() -> None:
    # python
    _py_0: FloatScalar = float(1234.0)
    _py_1: FloatScalar = 0.0
    # numpy
    _np_0: FloatScalar = np.float64(1234.0)
    # pytorch
    _pt_0: FloatScalar = pt.tensor([1234.0], dtype=pt.float64)


def type_complex_assignable() -> None:
    # python
    _py_0: ComplexScalar = complex(1, 2)
    _py_1: ComplexScalar = 0 + 0j
    # numpy
    _np_0: ComplexScalar = np.complex128(1 + 2j)
    # pytorch
    _pt_0: ComplexScalar = pt.tensor([1 + 2j], dtype=pt.complex128)
