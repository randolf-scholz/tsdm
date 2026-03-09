r"""Inspection tests for scalar types."""
# ruff: noqa: PLR2044

from datetime import datetime, timedelta
from typing import assert_type, reveal_type

from tsdm.experimental.types import (
    BoolScalar,
    ComplexScalar,
    DatetimeScalar,
    FloatScalar,
    IntScalar,
    SpanLikeScalar,
    TimedeltaScalar,
    TimeLikeScalar,
)

from .fixtures import SCALARS, types0d

type np_float = types0d.np.float
type np_int = types0d.np.int
type np_timedelta = types0d.np.timedelta
type np_datetime = types0d.np.datetime
type pd_timedelta = types0d.pd.timedelta
type pd_datetime = types0d.pd.datetime


class TestInferGenericType:
    def test_booleanscalar(self) -> None:
        # fmt: off
        def _view[T](_: BoolScalar[T]) -> T: ...  # type: ignore[empty-body]
        #
        reveal_type( _view(SCALARS.BOOL.PY) )
        reveal_type( _view(SCALARS.BOOL.NP) )
        reveal_type( _view(SCALARS.BOOL.PT) )
        # fmt: on

    def test_intscalar(self) -> None:
        # fmt: off
        def _view[T](_: IntScalar[T]) -> T: ...  # type: ignore[empty-body]
        #
        reveal_type( _view(SCALARS.INT.PY) )
        reveal_type( _view(SCALARS.INT.NP) )
        reveal_type( _view(SCALARS.INT.PT) )
        # fmt: on

    def test_floatscalar(self) -> None:
        # fmt: off
        def _view[T](_: FloatScalar[T]) -> T: ...  # type: ignore[empty-body]
        #
        reveal_type( _view(SCALARS.FLOAT.PY) )
        reveal_type( _view(SCALARS.FLOAT.NP) )
        reveal_type( _view(SCALARS.FLOAT.PT) )
        # fmt: on

    def test_complexscalar(self) -> None:
        # fmt: off
        def _view[T](_: ComplexScalar[T]) -> T: ...  # type: ignore[empty-body]
        #
        reveal_type( _view(SCALARS.COMPLEX.PY) )
        reveal_type( _view(SCALARS.COMPLEX.NP) )
        reveal_type( _view(SCALARS.COMPLEX.PT) )
        # fmt: on

    def test_spanlikescalar(self) -> None:
        # fmt: off
        def _view[SpanT](_: SpanLikeScalar[SpanT]) -> SpanT: ...  # type: ignore[empty-body]
        #
        reveal_type( _view(SCALARS.FLOAT.NP)     )
        reveal_type( _view(SCALARS.FLOAT.PY)     )
        reveal_type( _view(SCALARS.INT.NP)       )
        reveal_type( _view(SCALARS.INT.PY)       )
        reveal_type( _view(SCALARS.TIMEDELTA.NP) )
        reveal_type( _view(SCALARS.TIMEDELTA.PD) )
        reveal_type( _view(SCALARS.TIMEDELTA.PY) )
        # fmt: on

    def test_timelikescalar(self) -> None:
        # fmt: off
        def _view[  # type: ignore[empty-body]
            TimeT,
            SpanT,
            DualT: SpanLikeScalar,
        ](_: TimeLikeScalar[TimeT, SpanT, DualT]) -> tuple[TimeT, SpanT, DualT]: ...
        #
        reveal_type( _view(SCALARS.PY.FLOAT)    )
        reveal_type( _view(SCALARS.PY.INT)      )
        reveal_type( _view(SCALARS.PY.DATETIME) )
        reveal_type( _view(SCALARS.NP.FLOAT)    )
        reveal_type( _view(SCALARS.NP.INT)      )
        reveal_type( _view(SCALARS.NP.DATETIME) )
        reveal_type( _view(SCALARS.PD.DATETIME) )
        # fmt: on


class TestInferIdentity:
    def test_timelikescalar(self) -> None:
        # fmt: off
        def _id[DT: TimeLikeScalar](x: DT, /) -> DT: return x
        #
        assert_type(_id(SCALARS.DATETIME.NP), np_datetime)  # type: ignore[type-var]
        assert_type(_id(SCALARS.DATETIME.PD), pd_datetime)
        assert_type(_id(SCALARS.DATETIME.PY), datetime)
        assert_type(_id(SCALARS.FLOAT.NP), np_float)
        assert_type(_id(SCALARS.FLOAT.PY), float)
        assert_type(_id(SCALARS.INT.NP), np_int)
        assert_type(_id(SCALARS.INT.PY), int)
        # fmt: on

    def test_spanlikescalar(self) -> None:
        # fmt: off
        def _id[TD: SpanLikeScalar](x: TD, /) -> TD: return x
        #
        assert_type(_id(SCALARS.FLOAT.NP), np_float)
        assert_type(_id(SCALARS.FLOAT.PY), float)
        assert_type(_id(SCALARS.INT.NP), np_int)
        assert_type(_id(SCALARS.INT.PY), int)
        assert_type(_id(SCALARS.TIMEDELTA.NP), np_timedelta)
        assert_type(_id(SCALARS.TIMEDELTA.PD), pd_timedelta)
        assert_type(_id(SCALARS.TIMEDELTA.PY), timedelta)
        # fmt: on

    def test_datetimescalar(self) -> None:
        # fmt: off
        def _id[DT: DatetimeScalar](x: DT, /) -> DT: return x
        #
        assert_type(_id(SCALARS.DATETIME.NP), )  # type: ignore[type-var]
        assert_type(_id(SCALARS.DATETIME.PD), )
        assert_type(_id(SCALARS.DATETIME.PY), )
        # fmt: on

    def test_timedeltascalar(self) -> None:
        # fmt: off
        def _id[TD: TimedeltaScalar](x: TD, /) -> TD: return x
        #
        assert_type(_id(SCALARS.TIMEDELTA.NP), types0d.np.timedelta)  # type: ignore[type-var]
        assert_type(_id(SCALARS.TIMEDELTA.PD), types0d.pd.timedelta)
        assert_type(_id(SCALARS.TIMEDELTA.PY), types0d.py.timedelta)
        # fmt: on

    def test_complexscalar(self) -> None:
        # fmt: off
        def _id[T: ComplexScalar](x: T, /) -> T: return x
        #
        assert_type(_id(SCALARS.COMPLEX.PY), types0d.py.complex)
        assert_type(_id(SCALARS.COMPLEX.NP), types0d.np.complex)
        assert_type(_id(SCALARS.COMPLEX.PT), types0d.pt.complex)
        # fmt: on

    def test_floatscalar(self) -> None:
        # fmt: off
        def _id[T: FloatScalar](x: T, /) -> T: return x
        #
        assert_type(_id(SCALARS.FLOAT.PY), types0d.py.float)
        assert_type(_id(SCALARS.FLOAT.NP), types0d.np.float)
        assert_type(_id(SCALARS.FLOAT.PT), types0d.pt.float)
        # fmt: on

    def test_intscalar(self) -> None:
        # fmt: off
        def _id[T: IntScalar](x: T, /) -> T: return x
        #
        assert_type(_id(SCALARS.INT.PY), types0d.py.int)
        assert_type(_id(SCALARS.INT.NP), types0d.np.int)
        assert_type(_id(SCALARS.INT.PT), types0d.pt.int)
        # fmt: on

    def test_booleanscalar(self) -> None:
        # fmt: off
        def _id[T: BoolScalar](x: T, /) -> T: return x
        #
        assert_type(_id(SCALARS.BOOL.PY), types0d.py.bool)
        assert_type(_id(SCALARS.BOOL.NP), types0d.np.bool)
        assert_type(_id(SCALARS.BOOL.PT), types0d.pt.bool)
        # fmt: on
