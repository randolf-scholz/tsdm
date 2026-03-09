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

from .fixtures import SERIES, types0d


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
        _numpy__0: BooleanArray = SERIES.NP.BOOL
        _torch__0: BooleanArray = SERIES.PT.BOOL
        _pandas_0: BooleanArray = SERIES.PD_NP.BOOL
        _pandas_1: BooleanArray = SERIES.PD_PA.BOOL
        _polars_0: BooleanArray = SERIES.PL.BOOL
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: BooleanArray[Any] = SERIES.NP.BOOL
        _torch__0: BooleanArray[Any] = SERIES.PT.BOOL
        _pandas_0: BooleanArray[Any] = SERIES.PD_NP.BOOL
        _pandas_1: BooleanArray[Any] = SERIES.PD_PA.BOOL
        _polars_0: BooleanArray[Any] = SERIES.PL.BOOL
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: BooleanArray[bool] = SERIES.NP.BOOL
        _torch__0: BooleanArray[bool] = SERIES.PT.BOOL
        _pandas_0: BooleanArray[bool] = SERIES.PD_NP.BOOL
        _pandas_1: BooleanArray[bool] = SERIES.PD_PA.BOOL
        _polars_0: BooleanArray[bool] = SERIES.PL.BOOL
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: BooleanArray[types0d.np.bool] = SERIES.NP.BOOL
        _torch__0: BooleanArray[types0d.pt.bool] = SERIES.PT.BOOL
        _pandas_0: BooleanArray[types0d.pd.bool] = SERIES.PD_NP.BOOL
        _pandas_1: BooleanArray[types0d.pd.bool] = SERIES.PD_PA.BOOL
        _polars_0: BooleanArray[types0d.pl.bool] = SERIES.PL.BOOL
        # fmt: on


class TestIntAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray = SERIES.NP.INT
        _torch__0: IntegerArray = SERIES.PT.INT
        _pandas_0: IntegerArray = SERIES.PD_NP.INT
        _pandas_1: IntegerArray = SERIES.PD_PA.INT
        _polars_0: IntegerArray = SERIES.PL.INT
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray[Any] = SERIES.NP.INT
        _torch__0: IntegerArray[Any] = SERIES.PT.INT
        _pandas_0: IntegerArray[Any] = SERIES.PD_NP.INT
        _pandas_1: IntegerArray[Any] = SERIES.PD_PA.INT
        _polars_0: IntegerArray[Any] = SERIES.PL.INT
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray[int] = SERIES.NP.INT
        _torch__0: IntegerArray[int] = SERIES.PT.INT
        _pandas_0: IntegerArray[int] = SERIES.PD_NP.INT
        _pandas_1: IntegerArray[int] = SERIES.PD_PA.INT
        _polars_0: IntegerArray[int] = SERIES.PL.INT
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: IntegerArray[types0d.np.int] = SERIES.NP.INT
        _torch__0: IntegerArray[types0d.pt.int] = SERIES.PT.INT
        _pandas_0: IntegerArray[types0d.pd.int] = SERIES.PD_NP.INT
        _pandas_1: IntegerArray[types0d.pd.int] = SERIES.PD_PA.INT
        _polars_0: IntegerArray[types0d.pl.int] = SERIES.PL.INT
        # fmt: on


class TestFloatAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray = SERIES.NP.FLOAT
        _torch__0: FloatArray = SERIES.PT.FLOAT
        _pandas_0: FloatArray = SERIES.PD_NP.FLOAT
        _pandas_1: FloatArray = SERIES.PD_PA.FLOAT
        _polars_0: FloatArray = SERIES.PL.FLOAT
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray[Any] = SERIES.NP.FLOAT
        _torch__0: FloatArray[Any] = SERIES.PT.FLOAT
        _pandas_0: FloatArray[Any] = SERIES.PD_NP.FLOAT
        _pandas_1: FloatArray[Any] = SERIES.PD_PA.FLOAT
        _polars_0: FloatArray[Any] = SERIES.PL.FLOAT
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray[float] = SERIES.NP.FLOAT
        _torch__0: FloatArray[float] = SERIES.PT.FLOAT
        _pandas_0: FloatArray[float] = SERIES.PD_NP.FLOAT
        _pandas_1: FloatArray[float] = SERIES.PD_PA.FLOAT
        _polars_0: FloatArray[float] = SERIES.PL.FLOAT
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: FloatArray[types0d.np.float] = SERIES.NP.FLOAT
        _torch__0: FloatArray[types0d.pt.float] = SERIES.PT.FLOAT
        _pandas_0: FloatArray[types0d.pd.float] = SERIES.PD_NP.FLOAT
        _pandas_1: FloatArray[types0d.pd.float] = SERIES.PD_PA.FLOAT
        _polars_0: FloatArray[types0d.pl.float] = SERIES.PL.FLOAT
        # fmt: on


class TestComplexAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray = SERIES.NP.COMPLEX
        _torch__0: ComplexArray = SERIES.PT.COMPLEX
        _pandas_0: ComplexArray = SERIES.PD_NP.COMPLEX
        # _pandas_1: ComplexArray = SERIES.PD_PA.COMPLEX
        # _polars_0: ComplexArray = SERIES.PL.COMPLEX
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray[Any] = SERIES.NP.COMPLEX
        _torch__0: ComplexArray[Any] = SERIES.PT.COMPLEX
        _pandas_0: ComplexArray[Any] = SERIES.PD_NP.COMPLEX
        # _pandas_1: ComplexArray[Any] = SERIES.PD_PA.COMPLEX
        # _polars_0: ComplexArray[Any] = SERIES.PL.COMPLEX
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray[complex] = SERIES.NP.COMPLEX
        _torch__0: ComplexArray[complex] = SERIES.PT.COMPLEX
        _pandas_0: ComplexArray[complex] = SERIES.PD_NP.COMPLEX
        # _pandas_1: ComplexArray[complex] = SERIES.PD_PA.COMPLEX
        # _polars_0: ComplexArray[complex] = SERIES.PL.COMPLEX
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: ComplexArray[types0d.np.complex] = SERIES.NP.COMPLEX
        _torch__0: ComplexArray[types0d.pt.complex] = SERIES.PT.COMPLEX
        _pandas_0: ComplexArray[types0d.pd.complex] = SERIES.PD_NP.COMPLEX
        # _pandas_1: ComplexArray[types0d.pd.complex] = SERIES.PD_PA.COMPLEX
        # _polars_0: ComplexArray[types0d.pl.complex] = SERIES.PL.COMPLEX
        # fmt: on


class TestTimedeltaAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: SpanLikeArray = SERIES.NP.TIMEDELTA
        _pandas_0: SpanLikeArray = SERIES.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray = SERIES.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray = SERIES.PL.TIMEDELTA
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: SpanLikeArray[Any] = SERIES.NP.TIMEDELTA
        _pandas_0: SpanLikeArray[Any] = SERIES.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray[Any] = SERIES.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray[Any] = SERIES.PL.TIMEDELTA
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy__0: SpanLikeArray[timedelta] = SERIES.NP.TIMEDELTA  # INCOMPATIBLE
        _pandas_0: SpanLikeArray[timedelta] = SERIES.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray[timedelta] = SERIES.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray[timedelta] = SERIES.PL.TIMEDELTA
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: SpanLikeArray[types0d.np.timedelta] = SERIES.NP.TIMEDELTA
        _pandas_0: SpanLikeArray[types0d.pd.timedelta] = SERIES.PD_NP.TIMEDELTA
        _pandas_1: SpanLikeArray[types0d.pd.timedelta] = SERIES.PD_PA.TIMEDELTA
        _polars_0: SpanLikeArray[types0d.pl.timedelta] = SERIES.PL.TIMEDELTA
        # fmt: on


class TestDatetimeAssignable:
    def test_no_generic(self) -> None:
        # fmt: off
        _numpy__0: TimeLikeArray = SERIES.NP.DATETIME
        _pandas_0: TimeLikeArray = SERIES.PD_NP.DATETIME
        _pandas_1: TimeLikeArray = SERIES.PD_PA.DATETIME
        _polars_0: TimeLikeArray = SERIES.PL.DATETIME
        # fmt: on

    def test_any_generic(self) -> None:
        # fmt: off
        _numpy__0: TimeLikeArray[Any] = SERIES.NP.DATETIME
        _pandas_0: TimeLikeArray[Any] = SERIES.PD_NP.DATETIME
        _pandas_1: TimeLikeArray[Any] = SERIES.PD_PA.DATETIME
        _polars_0: TimeLikeArray[Any] = SERIES.PL.DATETIME
        # fmt: on

    def test_python_generic(self) -> None:
        # fmt: off
        # _numpy__0: TimeLikeArray[datetime] = SERIES.NP.DATETIME  # INCOMPATIBLE
        _pandas_0: TimeLikeArray[datetime] = SERIES.PD_NP.DATETIME
        _pandas_1: TimeLikeArray[datetime] = SERIES.PD_PA.DATETIME
        _polars_0: TimeLikeArray[datetime] = SERIES.PL.DATETIME
        # fmt: on

    def test_self_generic(self) -> None:
        # fmt: off
        _numpy__0: TimeLikeArray[types0d.np.datetime] = SERIES.NP.DATETIME
        _pandas_0: TimeLikeArray[types0d.pd.datetime] = SERIES.PD_NP.DATETIME
        _pandas_1: TimeLikeArray[types0d.pd.datetime] = SERIES.PD_PA.DATETIME
        _polars_0: TimeLikeArray[types0d.pl.datetime] = SERIES.PL.DATETIME
        # fmt: on


class TestInspection:
    def inspect_bool_array(self) -> None:
        def _view[T](_: BooleanArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(SERIES.NP.BOOL))
        reveal_type(_view(SERIES.PT.BOOL))
        reveal_type(_view(SERIES.PD_NP.BOOL))
        reveal_type(_view(SERIES.PD_PA.BOOL))
        reveal_type(_view(SERIES.PL.BOOL))
        # fmt: on

    def inspect_int_array(self) -> None:
        def _view[T](_: IntegerArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(SERIES.NP.INT))
        reveal_type(_view(SERIES.PT.INT))
        reveal_type(_view(SERIES.PD_NP.INT))
        reveal_type(_view(SERIES.PD_PA.INT))
        reveal_type(_view(SERIES.PL.INT))
        # fmt: on

    def inspect_float_array(self) -> None:
        def _view[T](_: FloatArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(SERIES.NP.FLOAT))
        reveal_type(_view(SERIES.PT.FLOAT))
        reveal_type(_view(SERIES.PD_NP.FLOAT))
        reveal_type(_view(SERIES.PD_PA.FLOAT))
        reveal_type(_view(SERIES.PL.FLOAT))
        # fmt: on

    def inspect_complex_array(self) -> None:
        def _view[T](_: ComplexArray[T]) -> T: ...

        # fmt: off
        reveal_type(_view(SERIES.NP.COMPLEX))
        reveal_type(_view(SERIES.PT.COMPLEX))
        reveal_type(_view(SERIES.PD_NP.COMPLEX))
        # reveal_type(_view(SERIES.PD_PA.COMPLEX))  # optional / unsupported
        # reveal_type(_view(SERIES.PL.COMPLEX))     # optional / unsupported
        # fmt: on

    def inspect_spanlike_array(self) -> None:
        def _view[SpanT](
            _: SpanLikeArray[SpanT],
        ) -> SpanT: ...

        # fmt: off
        reveal_type(_view(SERIES.NP.TIMEDELTA))
        reveal_type(_view(SERIES.NP.FLOAT))
        reveal_type(_view(SERIES.NP.INT))
        reveal_type(_view(SERIES.PD_NP.TIMEDELTA))
        reveal_type(_view(SERIES.PD_NP.FLOAT))
        reveal_type(_view(SERIES.PD_NP.INT))
        reveal_type(_view(SERIES.PD_NP.TIMEDELTA))
        reveal_type(_view(SERIES.PD_PA.FLOAT))
        reveal_type(_view(SERIES.PD_PA.INT))
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
        reveal_type(_view(SERIES.NP.DATETIME))
        reveal_type(_view(SERIES.NP.FLOAT))
        reveal_type(_view(SERIES.NP.INT))
        reveal_type(_view(SERIES.PD_NP.DATETIME))
        reveal_type(_view(SERIES.PD_NP.FLOAT))
        reveal_type(_view(SERIES.PD_NP.INT))
        reveal_type(_view(SERIES.PD_PA.DATETIME))
        reveal_type(_view(SERIES.PD_PA.FLOAT))
        reveal_type(_view(SERIES.PD_PA.INT))
        # fmt: on
