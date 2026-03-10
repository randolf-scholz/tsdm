r"""Static type tests for Numerical Array assignments."""
# pyright: reportUnusedFunction=false

from datetime import datetime, timedelta
from typing import Any, Never, reveal_type

from numerical_types import SpanLikeScalar, TimeLikeScalar
from numerical_types.arrays import (
    BooleanArray,
    ComplexArray,
    DatetimeArray,
    FloatArray,
    IntegerArray,
    SpanLikeArray,
    TimedeltaArray,
    TimeLikeArray,
)
from test_numerical_types.fixtures import ARRAYS1D, types0d


class TestContravariance:
    r"""We test that the compatible generics are contravariant."""

    def test_array(self) -> None:
        # fmt: off
        def _bool[T](x: BooleanArray[T]) -> BooleanArray[Never]:
            return x
        def _int[T](x: IntegerArray[T]) -> IntegerArray[Never]:
            return x
        def _float[T](x: FloatArray[T]) -> FloatArray[Never]:
            return x
        def _complex[T](x: ComplexArray[T]) -> ComplexArray[Never]:
            return x
        def _spanlike[
            SpanT: SpanLikeScalar,
        ](x: SpanLikeArray[SpanT]) -> SpanLikeArray[Never]:
            return x
        def _timedelta[
            SpanT: SpanLikeScalar,
        ](x: TimedeltaArray[SpanT]) -> TimedeltaArray[Never]:
            return x
        def _timelike[
            TimeT: TimeLikeScalar,
            SpanT: SpanLikeScalar,
        ](x: TimeLikeArray[TimeT, SpanT]) -> TimeLikeArray[Never, Never]:
            return x
        def _datetime[
            TimeT: TimeLikeScalar,
            SpanT: SpanLikeScalar,
        ](x: DatetimeArray[TimeT, SpanT]) -> DatetimeArray[Never, Never]:
            return x
        # fmt: on


class TestBooleanAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: BooleanArray = ARRAYS1D.NP.BOOL
        _torch__0: BooleanArray = ARRAYS1D.PT.BOOL
        _pandas_0: BooleanArray = ARRAYS1D.PD_NP.BOOL
        _pandas_1: BooleanArray = ARRAYS1D.PD_PA.BOOL
        _polars_0: BooleanArray = ARRAYS1D.PL.BOOL
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: BooleanArray[Any] = ARRAYS1D.NP.BOOL
        _torch__0: BooleanArray[Any] = ARRAYS1D.PT.BOOL
        _pandas_0: BooleanArray[Any] = ARRAYS1D.PD_NP.BOOL
        _pandas_1: BooleanArray[Any] = ARRAYS1D.PD_PA.BOOL
        _polars_0: BooleanArray[Any] = ARRAYS1D.PL.BOOL
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: BooleanArray[bool] = ARRAYS1D.NP.BOOL
        _torch__0: BooleanArray[bool] = ARRAYS1D.PT.BOOL
        _pandas_0: BooleanArray[bool] = ARRAYS1D.PD_NP.BOOL
        _pandas_1: BooleanArray[bool] = ARRAYS1D.PD_PA.BOOL
        _polars_0: BooleanArray[bool] = ARRAYS1D.PL.BOOL
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: BooleanArray[types0d.np.bool] = ARRAYS1D.NP.BOOL
        _torch__0: BooleanArray[types0d.pt.bool] = ARRAYS1D.PT.BOOL
        _pandas_0: BooleanArray[types0d.pd.bool] = ARRAYS1D.PD_NP.BOOL
        _pandas_1: BooleanArray[types0d.pd.bool] = ARRAYS1D.PD_PA.BOOL
        _polars_0: BooleanArray[types0d.pl.bool] = ARRAYS1D.PL.BOOL
        # fmt: on


class TestIntAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray = ARRAYS1D.NP.INT
        _torch__0: IntegerArray = ARRAYS1D.PT.INT
        _pandas_0: IntegerArray = ARRAYS1D.PD_NP.INT
        _pandas_1: IntegerArray = ARRAYS1D.PD_PA.INT
        _polars_0: IntegerArray = ARRAYS1D.PL.INT
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray[Any] = ARRAYS1D.NP.INT
        _torch__0: IntegerArray[Any] = ARRAYS1D.PT.INT
        _pandas_0: IntegerArray[Any] = ARRAYS1D.PD_NP.INT
        _pandas_1: IntegerArray[Any] = ARRAYS1D.PD_PA.INT
        _polars_0: IntegerArray[Any] = ARRAYS1D.PL.INT
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray[int] = ARRAYS1D.NP.INT
        _torch__0: IntegerArray[int] = ARRAYS1D.PT.INT
        _pandas_0: IntegerArray[int] = ARRAYS1D.PD_NP.INT
        _pandas_1: IntegerArray[int] = ARRAYS1D.PD_PA.INT
        _polars_0: IntegerArray[int] = ARRAYS1D.PL.INT
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray[types0d.np.int] = ARRAYS1D.NP.INT
        _torch__0: IntegerArray[types0d.pt.int] = ARRAYS1D.PT.INT
        _pandas_0: IntegerArray[types0d.pd.int] = ARRAYS1D.PD_NP.INT
        _pandas_1: IntegerArray[types0d.pd.int] = ARRAYS1D.PD_PA.INT
        _polars_0: IntegerArray[types0d.pl.int] = ARRAYS1D.PL.INT
        # fmt: on


class TestFloatAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray = ARRAYS1D.NP.FLOAT
        _torch__0: FloatArray = ARRAYS1D.PT.FLOAT
        _pandas_0: FloatArray = ARRAYS1D.PD_NP.FLOAT
        _pandas_1: FloatArray = ARRAYS1D.PD_PA.FLOAT
        _polars_0: FloatArray = ARRAYS1D.PL.FLOAT
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray[Any] = ARRAYS1D.NP.FLOAT
        _torch__0: FloatArray[Any] = ARRAYS1D.PT.FLOAT
        _pandas_0: FloatArray[Any] = ARRAYS1D.PD_NP.FLOAT
        _pandas_1: FloatArray[Any] = ARRAYS1D.PD_PA.FLOAT
        _polars_0: FloatArray[Any] = ARRAYS1D.PL.FLOAT
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray[float] = ARRAYS1D.NP.FLOAT
        _torch__0: FloatArray[float] = ARRAYS1D.PT.FLOAT
        _pandas_0: FloatArray[float] = ARRAYS1D.PD_NP.FLOAT
        _pandas_1: FloatArray[float] = ARRAYS1D.PD_PA.FLOAT
        _polars_0: FloatArray[float] = ARRAYS1D.PL.FLOAT
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray[types0d.np.float] = ARRAYS1D.NP.FLOAT
        _torch__0: FloatArray[types0d.pt.float] = ARRAYS1D.PT.FLOAT
        _pandas_0: FloatArray[types0d.pd.float] = ARRAYS1D.PD_NP.FLOAT
        _pandas_1: FloatArray[types0d.pd.float] = ARRAYS1D.PD_PA.FLOAT
        _polars_0: FloatArray[types0d.pl.float] = ARRAYS1D.PL.FLOAT
        # fmt: on


class TestComplexAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray = ARRAYS1D.NP.COMPLEX
        _torch__0: ComplexArray = ARRAYS1D.PT.COMPLEX
        _pandas_0: ComplexArray = ARRAYS1D.PD_NP.COMPLEX
        # _pandas_1: ComplexArray = ARRAYS1D.PD_PA.COMPLEX
        # _polars_0: ComplexArray = ARRAYS1D.PL.COMPLEX
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray[Any] = ARRAYS1D.NP.COMPLEX
        _torch__0: ComplexArray[Any] = ARRAYS1D.PT.COMPLEX
        _pandas_0: ComplexArray[Any] = ARRAYS1D.PD_NP.COMPLEX
        # _pandas_1: ComplexArray[Any] = ARRAYS1D.PD_PA.COMPLEX
        # _polars_0: ComplexArray[Any] = ARRAYS1D.PL.COMPLEX
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray[complex] = ARRAYS1D.NP.COMPLEX
        _torch__0: ComplexArray[complex] = ARRAYS1D.PT.COMPLEX
        _pandas_0: ComplexArray[complex] = ARRAYS1D.PD_NP.COMPLEX
        # _pandas_1: ComplexArray[complex] = ARRAYS1D.PD_PA.COMPLEX
        # _polars_0: ComplexArray[complex] = ARRAYS1D.PL.COMPLEX
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray[types0d.np.complex] = ARRAYS1D.NP.COMPLEX
        _torch__0: ComplexArray[types0d.pt.complex] = ARRAYS1D.PT.COMPLEX
        _pandas_0: ComplexArray[types0d.pd.complex] = ARRAYS1D.PD_NP.COMPLEX
        # _pandas_1: ComplexArray[types0d.pd.complex] = ARRAYS1D.PD_PA.COMPLEX
        # _polars_0: ComplexArray[types0d.pl.complex] = ARRAYS1D.PL.COMPLEX
        # fmt: on


class TestTimedeltaAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: SpanLikeArray = ARRAYS1D.NP.TIMEDELTA
        _pandas_0: SpanLikeArray = ARRAYS1D.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray = ARRAYS1D.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray = ARRAYS1D.PL.TIMEDELTA
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: SpanLikeArray[Any] = ARRAYS1D.NP.TIMEDELTA
        _pandas_0: SpanLikeArray[Any] = ARRAYS1D.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray[Any] = ARRAYS1D.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray[Any] = ARRAYS1D.PL.TIMEDELTA
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy__0: SpanLikeArray[timedelta] = ARRAYS1D.NP.TIMEDELTA  # INCOMPATIBLE
        _pandas_0: SpanLikeArray[timedelta] = ARRAYS1D.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray[timedelta] = ARRAYS1D.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray[timedelta] = ARRAYS1D.PL.TIMEDELTA
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: SpanLikeArray[types0d.np.timedelta] = ARRAYS1D.NP.TIMEDELTA
        _pandas_0: SpanLikeArray[types0d.pd.timedelta] = ARRAYS1D.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray[types0d.pd.timedelta] = ARRAYS1D.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray[types0d.pl.timedelta] = ARRAYS1D.PL.TIMEDELTA
        # fmt: on


class TestDatetimeAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: TimeLikeArray = ARRAYS1D.NP.DATETIME
        _pandas_0: TimeLikeArray = ARRAYS1D.PD_NP.DATETIME
        _pandas_1: TimeLikeArray = ARRAYS1D.PD_PA.DATETIME
        _polars_0: TimeLikeArray = ARRAYS1D.PL.DATETIME
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: TimeLikeArray[Any] = ARRAYS1D.NP.DATETIME
        _pandas_0: TimeLikeArray[Any] = ARRAYS1D.PD_NP.DATETIME
        _pandas_1: TimeLikeArray[Any] = ARRAYS1D.PD_PA.DATETIME
        _polars_0: TimeLikeArray[Any] = ARRAYS1D.PL.DATETIME
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy__0: TimeLikeArray[datetime] = ARRAYS1D.NP.DATETIME  # INCOMPATIBLE
        _pandas_0: TimeLikeArray[datetime] = ARRAYS1D.PD_NP.DATETIME
        _pandas_1: TimeLikeArray[datetime] = ARRAYS1D.PD_PA.DATETIME
        _polars_0: TimeLikeArray[datetime] = ARRAYS1D.PL.DATETIME
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: TimeLikeArray[types0d.np.datetime] = ARRAYS1D.NP.DATETIME
        _pandas_0: TimeLikeArray[types0d.pd.datetime] = ARRAYS1D.PD_NP.DATETIME
        _pandas_1: TimeLikeArray[types0d.pd.datetime] = ARRAYS1D.PD_PA.DATETIME
        _polars_0: TimeLikeArray[types0d.pl.datetime] = ARRAYS1D.PL.DATETIME
        # fmt: on


class TestInspection:
    def inspect_bool_array(self) -> None:
        def _view[T](_: BooleanArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(ARRAYS1D.NP.BOOL))
        reveal_type(_view(ARRAYS1D.PT.BOOL))
        reveal_type(_view(ARRAYS1D.PD_NP.BOOL))
        reveal_type(_view(ARRAYS1D.PD_PA.BOOL))
        reveal_type(_view(ARRAYS1D.PL.BOOL))
        # fmt: on

    def inspect_int_array(self) -> None:
        def _view[T](_: IntegerArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(ARRAYS1D.NP.INT))
        reveal_type(_view(ARRAYS1D.PT.INT))
        reveal_type(_view(ARRAYS1D.PD_NP.INT))
        reveal_type(_view(ARRAYS1D.PD_PA.INT))
        reveal_type(_view(ARRAYS1D.PL.INT))
        # fmt: on

    def inspect_float_array(self) -> None:
        def _view[T](_: FloatArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(ARRAYS1D.NP.FLOAT))
        reveal_type(_view(ARRAYS1D.PT.FLOAT))
        reveal_type(_view(ARRAYS1D.PD_NP.FLOAT))
        reveal_type(_view(ARRAYS1D.PD_PA.FLOAT))
        reveal_type(_view(ARRAYS1D.PL.FLOAT))
        # fmt: on

    def inspect_complex_array(self) -> None:
        def _view[T](_: ComplexArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(ARRAYS1D.NP.COMPLEX))
        reveal_type(_view(ARRAYS1D.PT.COMPLEX))
        reveal_type(_view(ARRAYS1D.PD_NP.COMPLEX))
        # reveal_type(_view(ARRAYS1D.PD_PA.COMPLEX))  # optional / unsupported
        # reveal_type(_view(ARRAYS1D.PL.COMPLEX))     # optional / unsupported
        # fmt: on

    def inspect_spanlike_array(self) -> None:
        def _view[SpanT](
            _: SpanLikeArray[SpanT],
        ) -> SpanT: ...

        # fmt: off
        reveal_type(_view(ARRAYS1D.NP.TIMEDELTA))
        reveal_type(_view(ARRAYS1D.NP.FLOAT))
        reveal_type(_view(ARRAYS1D.NP.INT))
        reveal_type(_view(ARRAYS1D.PD_NP.TIMEDELTA))
        reveal_type(_view(ARRAYS1D.PD_NP.FLOAT))
        reveal_type(_view(ARRAYS1D.PD_NP.INT))
        reveal_type(_view(ARRAYS1D.PD_NP.TIMEDELTA))
        reveal_type(_view(ARRAYS1D.PD_PA.FLOAT))
        reveal_type(_view(ARRAYS1D.PD_PA.INT))
        # fmt: on

    def inspect_timelike_array(self) -> None:
        def _view[
            TimeT: TimeLikeScalar,
            SpanT: SpanLikeScalar,
            DualT: SpanLikeArray,
        ](
            _: TimeLikeArray[TimeT, SpanT, DualT],
        ) -> tuple[TimeT, SpanT, DualT]: ...

        # fmt: off
        reveal_type(_view(ARRAYS1D.NP.DATETIME))
        reveal_type(_view(ARRAYS1D.NP.FLOAT))
        reveal_type(_view(ARRAYS1D.NP.INT))
        reveal_type(_view(ARRAYS1D.PD_NP.DATETIME))
        reveal_type(_view(ARRAYS1D.PD_NP.FLOAT))
        reveal_type(_view(ARRAYS1D.PD_NP.INT))
        reveal_type(_view(ARRAYS1D.PD_PA.DATETIME))
        reveal_type(_view(ARRAYS1D.PD_PA.FLOAT))
        reveal_type(_view(ARRAYS1D.PD_PA.INT))
        # fmt: on
