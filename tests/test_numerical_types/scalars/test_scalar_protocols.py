r"""Tests for `tsdm.types.scalars`."""

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from numerical_types.scalars import (
    AdditiveScalar,
    BoolScalar,
    ComplexScalar,
    DatetimeScalar,
    FloatScalar,
    IntScalar,
    OrderedScalar,
    SpanLikeScalar,
    TimedeltaScalar,
    TimeLikeScalar,
)
from test_utils import pytest_xfail
from tests.test_numerical_types.fixtures import (
    BOOL as PY_BOOL,
    COMPLEX as PY_COMPLEX,
    FLOAT as PY_FLOAT,
    INT as PY_INT,
    SCALARS,
)
from tsdm.testing import check_shared_interface

ORDERED_SCALARS: dict[str, OrderedScalar] = {
    "np_bool"      : SCALARS.NP.BOOL,
    "np_datetime"  : SCALARS.NP.DATETIME,
    "np_float"     : SCALARS.NP.FLOAT,
    "np_int"       : SCALARS.NP.INT,
    "np_timedelta" : SCALARS.NP.TIMEDELTA,
    "pd_timedelta" : SCALARS.PD.TIMEDELTA,
    "pd_timestamp" : SCALARS.PD.DATETIME,
    "pt_bool"      : SCALARS.PT.BOOL,
    "pt_float"     : SCALARS.PT.FLOAT,
    "pt_int"       : SCALARS.PT.INT,
    "py_bool"      : SCALARS.PY.BOOL,
    "py_bytes"     : SCALARS.PY.BYTES,
    "py_datetime"  : SCALARS.PY.DATETIME,
    "py_float"     : SCALARS.PY.FLOAT,
    "py_int"       : SCALARS.PY.INT,
    "py_string"    : SCALARS.PY.STRING,
    "py_timedelta" : SCALARS.PY.TIMEDELTA,
    "py_tuple"     : SCALARS.PY.TUPLE,
}  # fmt: skip
r"""Ordered scalars for testing."""

ADDITIVE_SCALARS: dict[str, AdditiveScalar] = {
    "np_complex"   : SCALARS.NP.COMPLEX,
    "np_float"     : SCALARS.NP.FLOAT,
    "np_int"       : SCALARS.NP.INT,
    "np_timedelta" : SCALARS.NP.TIMEDELTA,
    "pd_timedelta" : SCALARS.PD.TIMEDELTA,
    "pt_complex"   : SCALARS.PT.COMPLEX,
    "pt_float"     : SCALARS.PT.FLOAT,
    "pt_int"       : SCALARS.PT.INT,
    "py_complex"   : SCALARS.PY.COMPLEX,
    "py_float"     : SCALARS.PY.FLOAT,
    "py_int"       : SCALARS.PY.INT,
    "py_timedelta" : SCALARS.PY.TIMEDELTA,
}  # fmt: skip
r"""Additive scalars for testing."""

BOOLEAN_SCALARS: dict[str, BoolScalar[bool]] = {
    "np_bool"    : SCALARS.NP.BOOL,
    "np_literal" : SCALARS.NP.BOOL,
    "pt_bool"    : SCALARS.PT.BOOL,
    "py_bool"    : SCALARS.PY.BOOL,
    "py_literal" : SCALARS.PY.BOOL,
}  # fmt: skip
r"""Boolean scalars for testing."""

INT_SCALARS: dict[str, IntScalar[int]] = {
    "np_int"     : SCALARS.NP.INT,
    "pt_int"     : SCALARS.PT.INT,
    "py_int"     : SCALARS.PY.INT,
    "py_literal" : SCALARS.PY.INT,
}  # fmt: skip
r"""Integer scalars for testing."""

FLOAT_SCALARS: dict[str, FloatScalar[float]] = {
    "np_float"   : SCALARS.NP.FLOAT,
    "pt_float"   : SCALARS.PT.FLOAT,
    "py_float"   : SCALARS.PY.FLOAT,
    "py_literal" : SCALARS.PY.FLOAT,
}  # fmt: skip
r"""Float scalars for testing."""

COMPLEX_SCALARS: dict[str, ComplexScalar[complex]] = {
    "np_complex" : SCALARS.NP.COMPLEX,
    "pt_complex" : SCALARS.PT.COMPLEX,
    "py_complex" : SCALARS.PY.COMPLEX,
    "py_literal" : SCALARS.PY.COMPLEX,
}  # fmt: skip
r"""Complex scalars for testing."""

DATETIME_SCALARS: dict[str, DatetimeScalar] = {
    "np_time"  : SCALARS.DATETIME.NP,
    "pd_time"  : SCALARS.DATETIME.PD,
    "py_time"  : SCALARS.DATETIME.PY,
}  # fmt: skip
r"""Dictionary of datetime scalars."""

TIMEDELTA_SCALARS: dict[str, TimedeltaScalar] = {
    "np_time"  : SCALARS.NP.TIMEDELTA,
    "pd_time"  : SCALARS.PD.TIMEDELTA,
    "py_time"  : SCALARS.PY.TIMEDELTA,
}  # fmt: skip
r"""Dictionary of timedelta scalars."""

SPANLIKE_SCALARS: dict[str, SpanLikeScalar] = {
    "np_float" : np.float64(1.0),
    "np_int"   : np.int64(1),
    "np_time"  : np.timedelta64(1, "D"),
    "pd_time"  : pd.Timedelta("1D"),
    "py_float" : 1.0,
    "py_int"   : 1,
    "py_time"  : dt.timedelta(days=1),
}  # fmt: skip
r"""Dictionary of timedelta scalars."""

TIMELIKE_SCALARS: dict[str, TimeLikeScalar] = {
    "np_time"  : np.datetime64("2021-01-01"),
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
    SpanLikeScalar  : SPANLIKE_SCALARS,
    TimeLikeScalar : TIMELIKE_SCALARS,
}  # fmt: skip
r"""Test cases for scalar types."""


class TestAbstractScalars:
    @pytest.mark.parametrize("case", ORDERED_SCALARS)
    def test_ordered_scalar(self, case: str) -> None:
        value = ORDERED_SCALARS[case]
        assert isinstance(value, OrderedScalar)

        # fmt: off
        # test comparisons (self)
        assert isinstance(value == value, BoolScalar)  # __eq__
        assert isinstance(value != value, BoolScalar)  # __ne__
        assert isinstance(value < value, BoolScalar)  # __lt__
        assert isinstance(value <= value, BoolScalar)  # __le__
        assert isinstance(value > value, BoolScalar)  # __gt__
        assert isinstance(value >= value, BoolScalar)  # __ge__
        # value check comparisons
        assert value == value
        assert value <= value
        assert value >= value
        assert not (value != value)  # noqa: SIM202
        assert not (value > value)
        assert not (value < value)
        # fmt: on

    @pytest.mark.parametrize("case", ADDITIVE_SCALARS)
    def test_additive_scalar(self, case: str) -> None:
        value = ADDITIVE_SCALARS[case]
        cls = type(value)
        assert isinstance(value, AdditiveScalar)

        assert type(value + value) is cls  # __add__
        assert type(value - value) is cls  # __sub__

    @pytest.mark.parametrize("case", SPANLIKE_SCALARS)
    def test_spanlike_scalar(self, case: str) -> None:
        value = SPANLIKE_SCALARS[case]
        cls = type(value)
        assert isinstance(value, SpanLikeScalar)

        # fmt: off
        # region test unary operations
        assert type(abs(value)) is cls  # __abs__
        assert type(-value) is cls  # __neg__
        assert type(+value) is cls  # __pos__
        # endregion test unary operations
        # region test binary operations
        # test arithmetic operations (self)
        assert type(value + value) is cls  # __add__
        assert type(value - value) is cls  # __sub__
        assert isinstance(value / value, FloatScalar)  # __truediv__
        assert type(value % value) is cls  # __mod__
        assert isinstance(value // value, FloatScalar)  # __truediv__
        # test arithmetic operations (int)
        assert type(value * PY_INT) is cls  # __mul__
        assert type(PY_INT * value) is cls  # __rmul__
        # endregion test binary operations
        # region test comparisons
        # test comparisons (self)
        assert isinstance(value == value, BoolScalar)  # __eq__
        assert isinstance(value != value, BoolScalar)  # __ne__
        assert isinstance(value < value, BoolScalar)  # __lt__
        assert isinstance(value <= value, BoolScalar)  # __le__
        assert isinstance(value > value, BoolScalar)  # __gt__
        assert isinstance(value >= value, BoolScalar)  # __ge__
        # endregion test comparisons
        # fmt: on

    @pytest.mark.parametrize("case", TIMELIKE_SCALARS)
    def test_timelike_scalar(self, case: str) -> None:
        value = TIMELIKE_SCALARS[case]
        cls = type(value)
        assert isinstance(value, TimeLikeScalar)
        ZERO = value - value
        # fmt: off
        # region test binary operations
        assert type(value + ZERO) is cls  # __add__
        assert type(ZERO + value) is cls  # __radd__
        assert isinstance(value - value, SpanLikeScalar)  # __sub__
        # endregion test binary operations
        # region test comparisons
        assert isinstance(value > value, BoolScalar)
        assert isinstance(value < value, BoolScalar)
        assert isinstance(value >= value, BoolScalar)
        assert isinstance(value <= value, BoolScalar)
        assert isinstance(value == value, BoolScalar)
        assert isinstance(value != value, BoolScalar)
        # endregion test comparisons
        # fmt: on


class TestConcreteScalars:
    @pytest.mark.parametrize("case", BOOLEAN_SCALARS)
    def test_boolean_scalar[T: BoolScalar](self, case: str) -> None:
        value: T = BOOLEAN_SCALARS[case]
        cls = type(value)
        assert isinstance(value, BoolScalar)

        # fmt: off
        # region test unary operations
        # test conversions
        assert type(bool(value)) is bool  # __bool__
        assert type(int(value)) is int  # __int__
        assert type(float(value)) is float  # __float_
        # endregion test unary operations
        # region test binary operations
        # test boolean operators (self)
        assert type(value & value) is cls  # __and__
        assert type(value ^ value) is cls  # __xor__
        assert type(value | value) is cls  # __or__
        # test boolean operators (bool)
        assert type(value & PY_BOOL) is cls  # __and__
        assert type(value | PY_BOOL) is cls  # __or__
        assert type(value ^ PY_BOOL) is cls  # __xor__
        # test boolean operators (bool, reversed)
        assert type(PY_BOOL & value) is cls  # __rand__
        assert type(PY_BOOL | value) is cls  # __ror__
        assert type(PY_BOOL ^ value) is cls  # __rxor__
        # endregion test binary operations
        # region test comparisons
        # test comparisons (self)
        assert type(value == value) is cls  # __eq__
        assert type(value != value) is cls  # __ne__
        # test comparisons (bool)
        assert type(value == PY_BOOL) is cls  # __eq__
        assert type(value != PY_BOOL) is cls  # __ne__
        # test comparisons (int)
        assert type(value == PY_INT) is cls  # __eq__
        assert type(value != PY_INT) is cls  # __ne__
        # test comparisons (float)
        assert type(value == PY_FLOAT) is cls  # __eq__
        assert type(value != PY_FLOAT) is cls  # __ne__
        # test comparisons (complex)
        assert type(value == PY_COMPLEX) is cls  # __eq__
        assert type(value != PY_COMPLEX) is cls  # __ne__

        if not TYPE_CHECKING:
            # test comparisons (self)
            assert type(value < value) is cls  # __lt__
            assert type(value <= value) is cls  # __le__
            assert type(value > value) is cls  # __gt__
            assert type(value >= value) is cls  # __ge__
            # test comparisons (bool)
            assert type(value < PY_BOOL) is cls  # __lt__
            assert type(value <= PY_BOOL) is cls  # __le__
            assert type(value > PY_BOOL) is cls  # __gt__
            assert type(value >= PY_BOOL) is cls  # __ge__
            # test comparisons (int)
            assert type(value < PY_INT) is cls  # __lt__
            assert type(value <= PY_INT) is cls  # __le__
            assert type(value > PY_INT) is cls  # __gt__
            assert type(value >= PY_INT) is cls  # __ge__
            # test comparisons (float)
            assert type(value < PY_FLOAT) is cls  # __lt__
            assert type(value <= PY_FLOAT) is cls  # __le__
            assert type(value > PY_FLOAT) is cls  # __gt__
            assert type(value >= PY_FLOAT) is cls  # __ge__
        # endregion test comparisons
        # fmt: on

    @pytest.mark.parametrize("case", INT_SCALARS)
    def test_int_scalar[T](self, case: str) -> None:
        value: T = INT_SCALARS[case]
        cls = type(value)
        assert isinstance(value, IntScalar)

        # fmt: off
        # region test unary operations
        # test conversions
        assert type(bool(value)) is bool  # __bool__
        assert type(int(value)) is int  # __int__
        assert type(value.__index__()) is int  # __index__
        assert type(float(value)) is float  # __float_
        # test unary operations
        assert type(abs(value)) is cls  # __abs__
        assert type(-value) is cls  # __neg__
        assert type(+value) is cls  # __pos__
        # endregion test unary operations
        # region test binary operations
        # test arithmetic operations (self)
        assert type(value + value) is cls  # __add__
        assert type(value - value) is cls  # __sub__
        assert type(value * value) is cls  # __mul__
        assert type(value ** value) is cls  # __pow__
        assert type(value // value) is cls  # __floordiv__
        assert type(value % value) is cls  # __mod__
        # test arithmetic operations (int)
        assert type(value + PY_INT) is cls  # __add__
        assert type(value - PY_INT) is cls  # __sub__
        assert type(value * PY_INT) is cls  # __mul__
        assert type(value ** PY_INT) is cls  # __pow__
        assert type(value // PY_INT) is cls  # __floordiv__
        assert type(value % PY_INT) is cls  # __mod__
        # test arithmetic operations (int, reversed)
        assert type(PY_INT + value) is cls  # __radd__
        assert type(PY_INT - value) is cls  # __rsub__
        assert type(PY_INT * value) is cls  # __rmul__
        assert type(PY_INT ** value) is cls  # __rpow__
        assert type(PY_INT // value) is cls  # __rfloordiv__
        assert type(PY_INT % value) is cls  # __rmod__
        # test arithmetic operations (bool)
        assert type(value + PY_BOOL) is cls  # __add__
        # assert type(value -  BOOL) is cls   # __sub__
        assert type(value * PY_BOOL) is cls  # __mul__
        assert type(value ** PY_BOOL) is cls  # __pow__
        assert type(value // PY_BOOL) is cls  # __floordiv__
        assert type(value % PY_BOOL) is cls  # __mod__
        # test arithmetic operations (bool, reversed)
        assert type(PY_BOOL + value) is cls  # __radd__
        # assert type(BOOL -  value) is cls   # __rsub__
        assert type(PY_BOOL * value) is cls  # __rmul__
        assert type(PY_BOOL ** value) is cls  # __rpow__
        assert type(PY_BOOL // value) is cls  # __rfloordiv__
        assert type(PY_BOOL % value) is cls  # __rmod__
        # endregion test binary operations
        # region test comparisons
        # test comparisons (self)
        assert isinstance(value == value, BoolScalar)  # __eq__
        assert isinstance(value != value, BoolScalar)  # __ne__
        assert isinstance(value < value, BoolScalar)  # __lt__
        assert isinstance(value <= value, BoolScalar)  # __le__
        assert isinstance(value > value, BoolScalar)  # __gt__
        assert isinstance(value >= value, BoolScalar)  # __ge__
        # test comparisons (bool)
        assert isinstance(value == PY_BOOL, BoolScalar)  # __eq__
        assert isinstance(value != PY_BOOL, BoolScalar)  # __ne__
        assert isinstance(value < PY_BOOL, BoolScalar)  # __lt__
        assert isinstance(value <= PY_BOOL, BoolScalar)  # __le__
        assert isinstance(value > PY_BOOL, BoolScalar)  # __gt__
        assert isinstance(value >= PY_BOOL, BoolScalar)  # __ge__
        # test comparisons (int)
        assert isinstance(value == PY_INT, BoolScalar)  # __eq__
        assert isinstance(value != PY_INT, BoolScalar)  # __ne__
        assert isinstance(value < PY_INT, BoolScalar)  # __lt__
        assert isinstance(value <= PY_INT, BoolScalar)  # __le__
        assert isinstance(value > PY_INT, BoolScalar)  # __gt__
        assert isinstance(value >= PY_INT, BoolScalar)  # __ge__
        # test comparisons (float)
        assert isinstance(value == PY_FLOAT, BoolScalar)  # __eq__
        assert isinstance(value != PY_FLOAT, BoolScalar)  # __ne__
        assert isinstance(value < PY_FLOAT, BoolScalar)  # __lt__
        assert isinstance(value <= PY_FLOAT, BoolScalar)  # __le__
        assert isinstance(value > PY_FLOAT, BoolScalar)  # __gt__
        assert isinstance(value >= PY_FLOAT, BoolScalar)  # __ge__
        # test comparisons (complex)
        assert isinstance(value == PY_COMPLEX, BoolScalar)  # __eq__
        assert isinstance(value != PY_COMPLEX, BoolScalar)  # __ne__
        # endregion test comparisons
        # fmt: on

    @pytest.mark.parametrize("case", FLOAT_SCALARS)
    def test_float_scalar(self, case: str) -> None:
        value: FloatScalar = FLOAT_SCALARS[case]
        cls = type(value)
        assert isinstance(value, FloatScalar)

        # fmt: off
        # region test unary operations
        # test conversions
        assert type(bool(value)) is bool  # __bool__
        assert type(int(value)) is int  # __int__
        assert type(float(value)) is float  # __float_
        # test unary operations
        assert type(abs(value)) is cls  # __abs__
        assert type(-value) is cls  # __neg__
        assert type(+value) is cls  # __pos__
        # endregion test unary operations
        # region test binary operations
        # test arithmetic operations (self)
        assert type(value + value) is cls  # __add__
        assert type(value - value) is cls  # __sub__
        assert type(value * value) is cls  # __mul__
        assert type(value / value) is cls  # __truediv__
        assert type(value ** value) is cls  # __pow__
        assert type(value // value) is cls  # __floordiv__
        # test arithmetic operations (float)
        assert type(value + PY_FLOAT) is cls  # __add__
        assert type(value - PY_FLOAT) is cls  # __sub__
        assert type(value * PY_FLOAT) is cls  # __mul__
        assert type(value / PY_FLOAT) is cls  # __truediv__
        assert type(value ** PY_FLOAT) is cls  # __pow__
        assert type(value // PY_FLOAT) is cls  # __floordiv__
        # test arithmetic operations (float, reversed)
        assert type(PY_FLOAT + value) is cls  # __radd__
        assert type(PY_FLOAT - value) is cls  # __rsub__
        assert type(PY_FLOAT * value) is cls  # __rmul__
        assert type(PY_FLOAT / value) is cls  # __rtruediv__
        assert type(PY_FLOAT ** value) is cls  # __rpow__
        assert type(PY_FLOAT // value) is cls  # __rfloordiv__
        # endregion test binary operations
        # region test comparisons
        # test comparisons (self)
        assert isinstance(value == value, BoolScalar)  # __eq__
        assert isinstance(value != value, BoolScalar)  # __ne__
        assert isinstance(value < value, BoolScalar)  # __lt__
        assert isinstance(value <= value, BoolScalar)  # __le__
        assert isinstance(value > value, BoolScalar)  # __gt__
        assert isinstance(value >= value, BoolScalar)  # __ge__
        # test comparisons (bool)
        assert isinstance(value == PY_BOOL, BoolScalar)  # __eq__
        assert isinstance(value != PY_BOOL, BoolScalar)  # __ne__
        assert isinstance(value < PY_BOOL, BoolScalar)  # __lt__
        assert isinstance(value <= PY_BOOL, BoolScalar)  # __le__
        assert isinstance(value > PY_BOOL, BoolScalar)  # __gt__
        assert isinstance(value >= PY_BOOL, BoolScalar)  # __ge__
        # test comparisons (int)
        assert isinstance(value == PY_INT, BoolScalar)  # __eq__
        assert isinstance(value != PY_INT, BoolScalar)  # __ne__
        assert isinstance(value < PY_INT, BoolScalar)  # __lt__
        assert isinstance(value <= PY_INT, BoolScalar)  # __le__
        assert isinstance(value > PY_INT, BoolScalar)  # __gt__
        assert isinstance(value >= PY_INT, BoolScalar)  # __ge__
        # test comparisons (float)
        assert isinstance(value == PY_FLOAT, BoolScalar)  # __eq__
        assert isinstance(value != PY_FLOAT, BoolScalar)  # __ne__
        assert isinstance(value < PY_FLOAT, BoolScalar)  # __lt__
        assert isinstance(value <= PY_FLOAT, BoolScalar)  # __le__
        assert isinstance(value > PY_FLOAT, BoolScalar)  # __gt__
        assert isinstance(value >= PY_FLOAT, BoolScalar)  # __ge__
        # test comparisons (complex)
        assert isinstance(value == PY_COMPLEX, BoolScalar)  # __eq__
        assert isinstance(value != PY_COMPLEX, BoolScalar)  # __ne__
        # endregion test comparisons
        # fmt: on

    @pytest.mark.parametrize("name", COMPLEX_SCALARS)
    def test_complex_scalar(self, name: str) -> None:
        value = COMPLEX_SCALARS[name]
        cls = type(value)
        assert isinstance(value, ComplexScalar)

        # fmt: off
        # region test unary operations
        # test conversions
        assert type(bool(value)) is bool  # __bool__
        assert type(complex(value)) is complex  # __complex__
        # test unary operations
        assert isinstance(abs(value), FloatScalar)  # __abs__
        assert type(-value) is cls  # __neg__
        assert type(+value) is cls  # __pos__
        # endregion test unary operations
        # region test binary operations
        # test arithmetic operations (self)
        assert type(value + value) is cls  # __add__
        assert type(value - value) is cls  # __sub__
        assert type(value * value) is cls  # __mul__
        assert type(value / value) is cls  # __truediv__
        assert type(value ** value) is cls  # __pow__
        # test arithmetic operations (complex)
        assert type(value + PY_COMPLEX) is cls  # __add__
        assert type(value - PY_COMPLEX) is cls  # __sub__
        assert type(value * PY_COMPLEX) is cls  # __mul__
        assert type(value / PY_COMPLEX) is cls  # __truediv__
        assert type(value ** PY_COMPLEX) is cls  # __pow__
        # test arithmetic operations (complex, reversed)
        assert type(PY_COMPLEX + value) is cls  # __radd__
        assert type(PY_COMPLEX - value) is cls  # __rsub__
        assert type(PY_COMPLEX * value) is cls  # __rmul__
        assert type(PY_COMPLEX / value) is cls  # __rtruediv__
        assert type(PY_COMPLEX ** value) is cls  # __rpow__
        # test arithmetic operations (float)
        assert type(value + PY_FLOAT) is cls  # __add__
        assert type(value - PY_FLOAT) is cls  # __sub__
        assert type(value * PY_FLOAT) is cls  # __mul__
        assert type(value / PY_FLOAT) is cls  # __truediv__
        assert type(value ** PY_FLOAT) is cls  # __pow__
        # test arithmetic operations (float, reversed)
        assert type(PY_FLOAT + value) is cls  # __radd__
        assert type(PY_FLOAT - value) is cls  # __rsub__
        assert type(PY_FLOAT * value) is cls  # __rmul__
        assert type(PY_FLOAT / value) is cls  # __rtruediv__
        assert type(PY_FLOAT ** value) is cls  # __rpow__
        # endregion test binary operations
        # region test comparisons
        # test comparisons (self)
        assert isinstance(value == value, BoolScalar)  # __eq__
        assert isinstance(value != value, BoolScalar)  # __ne__
        # test comparisons (bool)
        assert isinstance(value == PY_BOOL, BoolScalar)  # __eq__
        assert isinstance(value != PY_BOOL, BoolScalar)  # __ne__
        # test comparisons (int)
        assert isinstance(value == PY_INT, BoolScalar)  # __eq__
        assert isinstance(value != PY_INT, BoolScalar)  # __ne__
        # test comparisons (float)
        assert isinstance(value == PY_FLOAT, BoolScalar)  # __eq__
        assert isinstance(value != PY_FLOAT, BoolScalar)  # __ne__
        # test comparisons (complex)
        assert isinstance(value == PY_COMPLEX, BoolScalar)  # __eq__
        assert isinstance(value != PY_COMPLEX, BoolScalar)  # __ne__
        # endregion test comparisons
        # fmt: on

    @pytest.mark.parametrize("name", SPANLIKE_SCALARS)
    def test_spanlike_scalar(self, name: str) -> None:
        value = SPANLIKE_SCALARS[name]
        assert isinstance(value, SpanLikeScalar)
        span_cls = type(value)

        # fmt: off
        # region test unary operations
        assert type(abs(value)) is span_cls  # __abs__
        assert type(-value) is span_cls  # __neg__
        assert type(+value) is span_cls  # __pos__
        # endregion test unary operations
        # region test binary operations
        # test arithmetic operations (self)
        assert type(value + value) is span_cls  # __add__
        assert type(value - value) is span_cls  # __sub__
        assert type(value * PY_INT) is span_cls  # __mul__
        assert type(PY_INT * value) is span_cls  # __rmul__
        assert type(value // PY_INT) is span_cls  # __floordiv__
        assert isinstance(value / value, FloatScalar)  # __truediv__
        assert isinstance(value // value, IntScalar | FloatScalar)  # __floordiv__
        assert type(value % value) is span_cls  # __mod__

        # endregion test binary operations
        # region test comparisons
        # test comparisons (self)
        assert isinstance(value == value, BoolScalar)  # __eq__
        assert isinstance(value != value, BoolScalar)  # __ne__
        assert isinstance(value < value, BoolScalar)  # __lt__
        assert isinstance(value <= value, BoolScalar)  # __le__
        assert isinstance(value > value, BoolScalar)  # __gt__
        assert isinstance(value >= value, BoolScalar)  # __ge__
        # endregion test comparisons
        if not TYPE_CHECKING:
            q, r = divmod(value, value)
            assert isinstance(q, IntScalar | FloatScalar)
            assert type(r) is span_cls
        # fmt: on

    @pytest.mark.parametrize("case", TIMEDELTA_SCALARS)
    def test_timedelta_scalar(self, case: str) -> None:
        value = TIMEDELTA_SCALARS[case]
        assert isinstance(value, TimedeltaScalar)
        span_cls = type(value)

        # fmt: off
        # region test comparisons
        assert isinstance(value == SCALARS.PY.TIMEDELTA, BoolScalar)
        assert isinstance(value != SCALARS.PY.TIMEDELTA, BoolScalar)
        assert isinstance(value >  SCALARS.PY.TIMEDELTA, BoolScalar)
        assert isinstance(value <  SCALARS.PY.TIMEDELTA, BoolScalar)
        assert isinstance(value >= SCALARS.PY.TIMEDELTA, BoolScalar)
        assert isinstance(value <= SCALARS.PY.TIMEDELTA, BoolScalar)
        # reversed comparisons
        assert isinstance(SCALARS.PY.TIMEDELTA == value, BoolScalar)
        assert isinstance(SCALARS.PY.TIMEDELTA != value, BoolScalar)
        assert isinstance(SCALARS.PY.TIMEDELTA >  value, BoolScalar)
        assert isinstance(SCALARS.PY.TIMEDELTA <  value, BoolScalar)
        assert isinstance(SCALARS.PY.TIMEDELTA >= value, BoolScalar)
        assert isinstance(SCALARS.PY.TIMEDELTA <= value, BoolScalar)
        # region test arithmetic operations
        with pytest_xfail(condition=case == "np_time"):
            # FIXME: https://github.com/numpy/numpy/issues/30985
            # addition with timedelta
            assert type(value + SCALARS.PY.TIMEDELTA) is span_cls  # __add__
            assert type(SCALARS.PY.TIMEDELTA + value) is span_cls  # __radd__
            # subtraction with timedelta
            assert type(SCALARS.PY.TIMEDELTA - value) is span_cls  # __rsub__
            assert type(value - SCALARS.PY.TIMEDELTA) is span_cls  # __sub__
            # division with timedelta
            assert isinstance(value / SCALARS.PY.TIMEDELTA, FloatScalar)  # __truediv__
            assert isinstance(SCALARS.PY.TIMEDELTA / value, FloatScalar)  # __rtruediv__
            # endregion test arithmetic operations
        # fmt: on

    @pytest.mark.parametrize("case", TIMELIKE_SCALARS)
    def test_timelike_scalar[T: TimeLikeScalar](self, case: str) -> None:
        value: T = TIMELIKE_SCALARS[case]
        span = value - value
        assert isinstance(value, TimeLikeScalar)
        assert isinstance(span, TimeLikeScalar)
        time_cls = type(value)
        span_cls = type(span)

        # fmt: off
        # region test binary operations
        assert type(value + span) is time_cls  # __add__
        assert type(span + value) is time_cls  # __radd__
        assert type(value - span) is time_cls  # __sub__
        assert type(value - value) is span_cls  # __sub__
        # endregion test binary operations
        # region test comparisons
        assert isinstance(value > value, BoolScalar)
        assert isinstance(value < value, BoolScalar)
        assert isinstance(value >= value, BoolScalar)
        assert isinstance(value <= value, BoolScalar)
        assert isinstance(value == value, BoolScalar)
        assert isinstance(value != value, BoolScalar)
        # endregion test comparisons
        # fmt: on

    @pytest.mark.parametrize("case", DATETIME_SCALARS)
    def test_datetime_scalar(self, case: str) -> None:
        value = DATETIME_SCALARS[case]
        assert isinstance(value, DatetimeScalar)
        assert isinstance(value - value, TimedeltaScalar)
        time_cls = type(value)

        # fmt: off
        # region test comparisons
        assert isinstance(value == SCALARS.PY.DATETIME, BoolScalar)
        assert isinstance(value != SCALARS.PY.DATETIME, BoolScalar)
        assert isinstance(value >  SCALARS.PY.DATETIME, BoolScalar)
        assert isinstance(value <  SCALARS.PY.DATETIME, BoolScalar)
        assert isinstance(value >= SCALARS.PY.DATETIME, BoolScalar)
        assert isinstance(value <= SCALARS.PY.DATETIME, BoolScalar)
        # reversed comparisons
        assert isinstance(SCALARS.PY.DATETIME == value, BoolScalar)
        assert isinstance(SCALARS.PY.DATETIME != value, BoolScalar)
        assert isinstance(SCALARS.PY.DATETIME >  value, BoolScalar)
        assert isinstance(SCALARS.PY.DATETIME <  value, BoolScalar)
        assert isinstance(SCALARS.PY.DATETIME >= value, BoolScalar)
        assert isinstance(SCALARS.PY.DATETIME <= value, BoolScalar)
        # region test arithmetic operations
        with pytest_xfail(condition=case == "np_time"):
            # FIXME: https://github.com/numpy/numpy/issues/30985
            # addition with timedelta
            assert type(value + SCALARS.PY.TIMEDELTA) is time_cls  # __add__
            assert type(SCALARS.PY.TIMEDELTA + value) is time_cls  # __radd__
            # subtraction with timedelta
            assert type(value - SCALARS.PY.TIMEDELTA) is time_cls  # __sub__
            # endregion test arithmetic operations
        # fmt: on


@pytest.mark.parametrize("protocol", TEST_TYPED_CASES)
def test_shared_interface(protocol: type) -> None:
    test_cases = TEST_TYPED_CASES[protocol]
    check_shared_interface(test_cases.values(), protocol, raise_on_extra=False)
