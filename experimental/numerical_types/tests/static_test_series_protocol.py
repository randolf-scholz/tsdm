# pyright: reportUnusedFunction=false


from typing import Never

from experimental.numerical_types import (
    BooleanArray,
    ComplexArray,
    DatetimeArray,
    FloatArray,
    IntegerArray,
    SpanLikeArray,
    TimedeltaArray,
    TimeLikeArray,
)
from experimental.numerical_types import (
    BooleanSeries,
    ComplexSeries,
    DatetimeSeries,
    FloatSeries,
    IntegerSeries,
    SpanLikeSeries,
    TimedeltaSeries,
    TimeLikeSeries,
)


def test_upcasting() -> None:
    # fmt: off
    def _bool[T](b: BooleanSeries[T]) -> BooleanArray[T]: return b
    def _int[T](i: IntegerSeries[T]) -> IntegerArray[T]: return i
    def _float[T](f: FloatSeries[T]) -> FloatArray[T]: return f
    def _complex[T](c: ComplexSeries[T]) -> ComplexArray[T]: return c
    def _spanlike[T](s: SpanLikeSeries[T]) -> SpanLikeArray[T]: return s
    def _timedelta[T](t: TimedeltaSeries[T]) -> TimedeltaArray[T]: return t
    def _timelike[TimeT, DualT](t: TimeLikeSeries[TimeT, DualT]) -> TimeLikeArray[TimeT, DualT]: return t
    def _datetime[TimeT, DualT](d: DatetimeSeries[TimeT, DualT]) -> DatetimeArray[TimeT, DualT]: return d
    # fmt: on


class TestContravariance:
    r"""We test that the compatible generics are contravariant."""

    def test_series(self) -> None:
        # fmt: off
        def _bool[T](x: BooleanSeries[T]) -> BooleanSeries[Never]:
            return x
        def _int[T](x: IntegerSeries[T]) -> IntegerSeries[Never]:
            return x
        def _float[T](x: FloatSeries[T]) -> FloatSeries[Never]:
            return x
        def _complex[T](x: ComplexSeries[T]) -> ComplexSeries[Never]:
            return x
        def _spanlike[SpanT](x: SpanLikeSeries[SpanT]) -> SpanLikeSeries[Never]:
            return x
        def _timedelta[SpanT](x: TimedeltaSeries[SpanT]) -> TimedeltaSeries[Never]:
            return x
        def _timelike[TimeT, SpanT](x: TimeLikeSeries[TimeT, SpanT]) -> TimeLikeSeries[Never, Never]:
            return x
        def _datetime[TimeT, SpanT](x: DatetimeSeries[TimeT, SpanT]) -> DatetimeSeries[Never, Never]:
            return x
        # fmt: on
