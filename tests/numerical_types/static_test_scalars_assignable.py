r"""Tests for `tsdm.types.scalars`."""
# pyright: reportUnusedFunction=false

from datetime import datetime, timedelta
from typing import Any, Never

from tsdm.experimental.types.scalars import (
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

from .fixtures import SCALARS, types0d as t


class TestBooleanAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _python_0: BoolScalar = SCALARS.PY.BOOL
        _python_2: BoolScalar = SCALARS.PY.TRUE
        _python_3: BoolScalar = SCALARS.PY.FALSE
        _numpy_0:  BoolScalar = SCALARS.NP.BOOL
        _numpy_1:  BoolScalar = SCALARS.NP.BOOL
        _numpy_2:  BoolScalar = SCALARS.NP.BOOL
        _torch_0:  BoolScalar = SCALARS.PT.BOOL
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _python_0: BoolScalar[Any] = SCALARS.PY.BOOL
        _python_2: BoolScalar[Any] = SCALARS.PY.TRUE
        _python_3: BoolScalar[Any] = SCALARS.PY.FALSE
        _numpy_0:  BoolScalar[Any] = SCALARS.NP.BOOL
        _numpy_1:  BoolScalar[Any] = SCALARS.NP.BOOL
        _numpy_2:  BoolScalar[Any] = SCALARS.NP.BOOL
        _torch_0:  BoolScalar[Any] = SCALARS.PT.BOOL
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _python_0: BoolScalar[bool] = SCALARS.PY.BOOL
        _python_2: BoolScalar[bool] = SCALARS.PY.TRUE
        _python_3: BoolScalar[bool] = SCALARS.PY.FALSE
        _numpy_0:  BoolScalar[bool] = SCALARS.NP.BOOL
        _numpy_1:  BoolScalar[bool] = SCALARS.NP.BOOL
        _numpy_2:  BoolScalar[bool] = SCALARS.NP.BOOL
        _torch_0:  BoolScalar[bool] = SCALARS.PT.BOOL
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _python_0: BoolScalar[t.py.bool] = SCALARS.PY.BOOL
        _python_2: BoolScalar[t.py.bool] = SCALARS.PY.TRUE
        _python_3: BoolScalar[t.py.bool] = SCALARS.PY.FALSE
        _numpy_0:  BoolScalar[t.np.bool] = SCALARS.NP.BOOL
        _numpy_1:  BoolScalar[t.np.bool] = SCALARS.NP.BOOL
        _numpy_2:  BoolScalar[t.np.bool] = SCALARS.NP.BOOL
        _torch_0:  BoolScalar[t.pt.bool] = SCALARS.PT.BOOL
        # fmt: on


class TestIntAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _python_0: IntScalar = SCALARS.PY.INT
        _python_1: IntScalar = SCALARS.PY.ONE
        _python_2: IntScalar = SCALARS.PY.ZERO
        _numpy_0:  IntScalar = SCALARS.NP.INT
        _torch_0:  IntScalar = SCALARS.PT.INT
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _python_0: IntScalar[Any] = SCALARS.PY.INT
        _python_1: IntScalar[Any] = SCALARS.PY.ONE
        _python_2: IntScalar[Any] = SCALARS.PY.ZERO
        _numpy_0:  IntScalar[Any] = SCALARS.NP.INT
        _torch_0:  IntScalar[Any] = SCALARS.PT.INT
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _python_0: IntScalar[int] = SCALARS.PY.INT
        _python_1: IntScalar[int] = SCALARS.PY.ONE
        _python_2: IntScalar[int] = SCALARS.PY.ZERO
        _numpy_0:  IntScalar[int] = SCALARS.NP.INT
        _torch_0:  IntScalar[int] = SCALARS.PT.INT
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _python_0: IntScalar[t.py.int] = SCALARS.PY.INT
        _python_1: IntScalar[t.py.int] = SCALARS.PY.ONE
        _python_2: IntScalar[t.py.int] = SCALARS.PY.ZERO
        _numpy_0:  IntScalar[t.np.int] = SCALARS.NP.INT
        _torch_0:  IntScalar[t.pt.int] = SCALARS.PT.INT
        # fmt: on


class TestFloatAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _python_0: FloatScalar = SCALARS.PY.FLOAT
        _numpy_0:  FloatScalar = SCALARS.NP.FLOAT
        _torch_0:  FloatScalar = SCALARS.PT.FLOAT
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _python_0: FloatScalar[Any] = SCALARS.PY.FLOAT
        _numpy_0:  FloatScalar[Any] = SCALARS.NP.FLOAT
        _torch_0:  FloatScalar[Any] = SCALARS.PT.FLOAT
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _python_0: FloatScalar[float] = SCALARS.PY.FLOAT
        _numpy_0:  FloatScalar[float] = SCALARS.NP.FLOAT
        _torch_0:  FloatScalar[float] = SCALARS.PT.FLOAT
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _python_0: FloatScalar[t.py.float] = SCALARS.PY.FLOAT
        _numpy_0:  FloatScalar[t.np.float] = SCALARS.NP.FLOAT
        _torch_0:  FloatScalar[t.pt.float] = SCALARS.PT.FLOAT
        # fmt: on


class TestComplexAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _python_0: ComplexScalar = SCALARS.PY.COMPLEX
        _numpy_0:  ComplexScalar = SCALARS.NP.COMPLEX
        _torch_0:  ComplexScalar = SCALARS.PT.COMPLEX
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _python_0: ComplexScalar[Any] = SCALARS.PY.COMPLEX
        _numpy_0:  ComplexScalar[Any] = SCALARS.NP.COMPLEX
        _torch_0:  ComplexScalar[Any] = SCALARS.PT.COMPLEX
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _python_0: ComplexScalar[complex] = SCALARS.PY.COMPLEX
        _numpy_0:  ComplexScalar[complex] = SCALARS.NP.COMPLEX
        _torch_0:  ComplexScalar[complex] = SCALARS.PT.COMPLEX
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _python_0: ComplexScalar[t.py.complex] = SCALARS.PY.COMPLEX
        _numpy_0:  ComplexScalar[t.np.complex] = SCALARS.NP.COMPLEX
        _torch_0:  ComplexScalar[t.pt.complex] = SCALARS.PT.COMPLEX
        # fmt: on


class TestTimeLikeAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        # FIXME: https://github.com/numpy/numpy/issues/28257
        _numpy_0 : TimeLikeScalar = SCALARS.NP.DATETIME  # type: ignore[assignment]
        _numpy_1 : TimeLikeScalar = SCALARS.NP.INT
        _numpy_2 : TimeLikeScalar = SCALARS.NP.FLOAT
        _python_0: TimeLikeScalar = SCALARS.PY.DATETIME
        _python_1: TimeLikeScalar = SCALARS.PY.INT
        _python_2: TimeLikeScalar = SCALARS.PY.FLOAT
        _pandas_0: TimeLikeScalar = SCALARS.PD.DATETIME
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        # FIXME: https://github.com/numpy/numpy/issues/28257
        _numpy_0 : TimeLikeScalar[Any] = SCALARS.NP.DATETIME  # type: ignore[assignment]
        _numpy_1 : TimeLikeScalar[Any] = SCALARS.NP.INT
        _numpy_2 : TimeLikeScalar[Any] = SCALARS.NP.FLOAT
        _python_0: TimeLikeScalar[Any] = SCALARS.PY.DATETIME
        _python_1: TimeLikeScalar[Any] = SCALARS.PY.INT
        _python_2: TimeLikeScalar[Any] = SCALARS.PY.FLOAT
        _pandas_0: TimeLikeScalar[Any] = SCALARS.PD.DATETIME
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy_0 : TimeLikeScalar[datetime, timedelta] = SCALARS.NP.DATETIME  # INCOMPATIBLE!
        _numpy_1 : TimeLikeScalar[float   , float    ] = SCALARS.NP.FLOAT
        _numpy_2 : TimeLikeScalar[int     , int      ] = SCALARS.NP.INT
        _python_0: TimeLikeScalar[datetime, timedelta] = SCALARS.PY.DATETIME
        _python_1: TimeLikeScalar[float   , float    ] = SCALARS.PY.FLOAT
        _python_2: TimeLikeScalar[int     , int      ] = SCALARS.PY.INT
        _pandas_0: TimeLikeScalar[datetime, timedelta] = SCALARS.PD.DATETIME
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        # FIXME: https://github.com/numpy/numpy/issues/28257
        _numpy_0 : TimeLikeScalar[t.np.datetime] = SCALARS.NP.DATETIME  # type: ignore[assignment]
        _numpy_1 : TimeLikeScalar[t.np.int     ] = SCALARS.NP.INT
        _numpy_2 : TimeLikeScalar[t.np.float   ] = SCALARS.NP.FLOAT
        _python_0: TimeLikeScalar[t.py.datetime] = SCALARS.PY.DATETIME
        _python_1: TimeLikeScalar[t.py.int     ] = SCALARS.PY.INT
        _python_2: TimeLikeScalar[t.py.float   ] = SCALARS.PY.FLOAT
        _pandas_0: TimeLikeScalar[t.pd.datetime] = SCALARS.PD.DATETIME
        # fmt: on


class TestDatetimeAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        # _numpy_0: DatetimeScalar = SCALARS.NP.DATETIME  # INCOMPATIBLE!
        _pandas_0: DatetimeScalar = SCALARS.PD.DATETIME
        _python_0: DatetimeScalar = SCALARS.PY.DATETIME
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        # _numpy_0: DatetimeScalar[Any] = SCALARS.NP.DATETIME  # INCOMPATIBLE!
        _pandas_0: DatetimeScalar[Any] = SCALARS.PD.DATETIME
        _python_0: DatetimeScalar[Any] = SCALARS.PY.DATETIME
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy_0: DatetimeScalar[datetime] = SCALARS.NP.DATETIME  # INCOMPATIBLE!
        _pandas_0: DatetimeScalar[datetime] = SCALARS.PD.DATETIME
        _python_0: DatetimeScalar[datetime] = SCALARS.PY.DATETIME
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        # _numpy_0: DatetimeScalar[t.np.datetime] = SCALARS.NP.DATETIME  # INCOMPATIBLE!
        _pandas_0: DatetimeScalar[t.pd.datetime] = SCALARS.PD.DATETIME
        _python_0: DatetimeScalar[t.py.datetime] = SCALARS.PY.DATETIME
        # fmt: on


class TestOrderedAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy_0 : OrderedScalar = SCALARS.NP.INT
        _numpy_1 : OrderedScalar = SCALARS.NP.FLOAT
        _numpy_2 : OrderedScalar = SCALARS.NP.TIMEDELTA
        _numpy_3 : OrderedScalar = SCALARS.NP.DATETIME
        _pandas_0: OrderedScalar = SCALARS.PD.TIMEDELTA
        _pandas_1: OrderedScalar = SCALARS.PD.DATETIME
        _torch_0 : OrderedScalar = SCALARS.PT.INT
        _torch_1 : OrderedScalar = SCALARS.PT.FLOAT
        _python_0: OrderedScalar = SCALARS.PY.INT
        _python_1: OrderedScalar = SCALARS.PY.FLOAT
        _python_2: OrderedScalar = SCALARS.PY.TIMEDELTA
        _python_3: OrderedScalar = SCALARS.PY.DATETIME
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy_0 : OrderedScalar[Any] = SCALARS.NP.INT
        _numpy_1 : OrderedScalar[Any] = SCALARS.NP.FLOAT
        _numpy_2 : OrderedScalar[Any] = SCALARS.NP.TIMEDELTA
        _numpy_3 : OrderedScalar[Any] = SCALARS.NP.DATETIME
        _pandas_0: OrderedScalar[Any] = SCALARS.PD.TIMEDELTA
        _pandas_1: OrderedScalar[Any] = SCALARS.PD.DATETIME
        _torch_0 : OrderedScalar[Any] = SCALARS.PT.INT
        _torch_1 : OrderedScalar[Any] = SCALARS.PT.FLOAT
        _python_0: OrderedScalar[Any] = SCALARS.PY.INT
        _python_1: OrderedScalar[Any] = SCALARS.PY.FLOAT
        _python_2: OrderedScalar[Any] = SCALARS.PY.TIMEDELTA
        _python_3: OrderedScalar[Any] = SCALARS.PY.DATETIME
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy_0 : OrderedScalar[int      ] = SCALARS.NP.INT
        _numpy_1 : OrderedScalar[float    ] = SCALARS.NP.FLOAT
        _numpy_2 : OrderedScalar[datetime ] = SCALARS.NP.DATETIME
        _numpy_3 : OrderedScalar[timedelta] = SCALARS.NP.TIMEDELTA
        _pandas_0: OrderedScalar[timedelta] = SCALARS.PD.TIMEDELTA
        _pandas_1: OrderedScalar[datetime ] = SCALARS.PD.DATETIME
        _torch_0 : OrderedScalar[int      ] = SCALARS.PT.INT
        _torch_1 : OrderedScalar[float    ] = SCALARS.PT.FLOAT
        _python_0: OrderedScalar[int      ] = SCALARS.PY.INT
        _python_1: OrderedScalar[float    ] = SCALARS.PY.FLOAT
        _python_2: OrderedScalar[timedelta] = SCALARS.PY.TIMEDELTA
        _python_3: OrderedScalar[datetime ] = SCALARS.PY.DATETIME
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy_0 : OrderedScalar[t.np.int      ] = SCALARS.NP.INT
        _numpy_1 : OrderedScalar[t.np.float    ] = SCALARS.NP.FLOAT
        _numpy_2 : OrderedScalar[t.np.datetime ] = SCALARS.NP.DATETIME
        _numpy_3 : OrderedScalar[t.np.timedelta] = SCALARS.NP.TIMEDELTA
        _pandas_0: OrderedScalar[t.pd.timedelta] = SCALARS.PD.TIMEDELTA
        _pandas_1: OrderedScalar[t.pd.datetime ] = SCALARS.PD.DATETIME
        _torch_0 : OrderedScalar[t.pt.int      ] = SCALARS.PT.INT
        _torch_1 : OrderedScalar[t.pt.float    ] = SCALARS.PT.FLOAT
        _python_0: OrderedScalar[t.py.int      ] = SCALARS.PY.INT
        _python_1: OrderedScalar[t.py.float    ] = SCALARS.PY.FLOAT
        _python_2: OrderedScalar[t.py.timedelta] = SCALARS.PY.TIMEDELTA
        _python_3: OrderedScalar[t.py.datetime ] = SCALARS.PY.DATETIME
        # fmt: on


class TestSpanLikeAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy_0 : SpanLikeScalar = SCALARS.NP.TIMEDELTA
        _numpy_1 : SpanLikeScalar = SCALARS.NP.INT
        _numpy_2 : SpanLikeScalar = SCALARS.NP.FLOAT
        _python_0: SpanLikeScalar = SCALARS.PY.TIMEDELTA
        _python_1: SpanLikeScalar = SCALARS.PY.INT
        _python_2: SpanLikeScalar = SCALARS.PY.FLOAT
        _pandas_0: SpanLikeScalar = SCALARS.PD.TIMEDELTA
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy_0 : SpanLikeScalar[Any] = SCALARS.NP.TIMEDELTA
        _numpy_1 : SpanLikeScalar[Any] = SCALARS.NP.INT
        _numpy_2 : SpanLikeScalar[Any] = SCALARS.NP.FLOAT
        _python_0: SpanLikeScalar[Any] = SCALARS.PY.TIMEDELTA
        _python_1: SpanLikeScalar[Any] = SCALARS.PY.INT
        _python_2: SpanLikeScalar[Any] = SCALARS.PY.FLOAT
        _pandas_0: SpanLikeScalar[Any] = SCALARS.PD.TIMEDELTA
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy_0 : SpanLikeScalar[timedelta] = SCALARS.NP.TIMEDELTA  # INCOMPATIBLE!
        _numpy_1 : SpanLikeScalar[int]       = SCALARS.NP.INT
        _numpy_2 : SpanLikeScalar[float]     = SCALARS.NP.FLOAT
        _python_0: SpanLikeScalar[timedelta] = SCALARS.PY.TIMEDELTA
        _python_1: SpanLikeScalar[int]       = SCALARS.PY.INT
        _python_2: SpanLikeScalar[float]     = SCALARS.PY.FLOAT
        _pandas_0: SpanLikeScalar[timedelta] = SCALARS.PD.TIMEDELTA
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy_0 : SpanLikeScalar[t.np.timedelta] = SCALARS.NP.TIMEDELTA
        _numpy_1 : SpanLikeScalar[t.np.int]       = SCALARS.NP.INT
        _numpy_2 : SpanLikeScalar[t.np.float]     = SCALARS.NP.FLOAT
        _python_0: SpanLikeScalar[t.py.timedelta] = SCALARS.PY.TIMEDELTA
        _python_1: SpanLikeScalar[t.py.int]       = SCALARS.PY.INT
        _python_2: SpanLikeScalar[t.py.float]     = SCALARS.PY.FLOAT
        _pandas_0: SpanLikeScalar[t.pd.timedelta] = SCALARS.PD.TIMEDELTA
        # fmt: on


class TestTimedeltaAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        # _numpy_0 : TimedeltaScalar = SCALARS.NP.TIMEDELTA  # INCOMPATIBLE!
        _pandas_0: TimedeltaScalar = SCALARS.PD.TIMEDELTA
        _python_0: TimedeltaScalar = SCALARS.PY.TIMEDELTA
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        # _numpy_0 : TimedeltaScalar[Any] = SCALARS.NP.TIMEDELTA  # INCOMPATIBLE!
        _pandas_0: TimedeltaScalar[Any] = SCALARS.PD.TIMEDELTA
        _python_0: TimedeltaScalar[Any] = SCALARS.PY.TIMEDELTA
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy_0 : TimedeltaScalar[timedelta] = SCALARS.NP.TIMEDELTA  # INCOMPATIBLE!
        _pandas_0: TimedeltaScalar[timedelta] = SCALARS.PD.TIMEDELTA
        _python_0: TimedeltaScalar[timedelta] = SCALARS.PY.TIMEDELTA
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        # _numpy_0 : TimedeltaScalar[t.np.timedelta] = SCALARS.NP.TIMEDELTA  # INCOMPATIBLE!
        _pandas_0: TimedeltaScalar[t.pd.timedelta] = SCALARS.PD.TIMEDELTA
        _python_0: TimedeltaScalar[t.py.timedelta] = SCALARS.PY.TIMEDELTA
        # fmt: on


class TestContravariance:
    r"""We test that the compatible generics are contravariant."""

    def test(self) -> None:
        # fmt: off
        def _bool[T](x: BoolScalar[T]) -> BoolScalar[Never]:
            return x
        def _int[T](x: IntScalar[T]) -> IntScalar[Never]:
            return x
        def _float[T](x: FloatScalar[T]) -> FloatScalar[Never]:
            return x
        def _complex[T](x: ComplexScalar[T]) -> ComplexScalar[Never]:
            return x
        def _spanlike[SpanT](x: SpanLikeScalar[SpanT]) -> SpanLikeScalar[Never]:
            return x
        def _timedelta[SpanT](x: TimedeltaScalar[SpanT]) -> TimedeltaScalar[Never]:
            return x
        def _timelike[TimeT, SpanT](x: TimeLikeScalar[TimeT, SpanT]) -> TimeLikeScalar[Never, Never]:
            return x
        def _datetime[TimeT, SpanT](x: DatetimeScalar[TimeT, SpanT]) -> DatetimeScalar[Never, Never]:
            return x
        # fmt: on
