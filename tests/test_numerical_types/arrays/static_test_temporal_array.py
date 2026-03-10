import datetime as dt

from numerical_types import SpanLikeArray, TimeLikeArray
from test_numerical_types.fixtures import ARRAYS1D


class TestTimeLikeArrayAssignments:
    r"""Static type tests for TimeLikeArray assignments."""

    def test_float(self) -> None:
        # fmt: off
        _np_0: TimeLikeArray[float] = ARRAYS1D.NP.FLOAT
        _pd_0: TimeLikeArray[float] = ARRAYS1D.PD_NP.FLOAT
        _pd_1: TimeLikeArray[float] = ARRAYS1D.PD_PA.FLOAT
        _pl_0: TimeLikeArray[float] = ARRAYS1D.PL.FLOAT
        # fmt: on

    def test_int(self) -> None:
        # fmt: off
        _np_0: TimeLikeArray[int] = ARRAYS1D.NP.INT
        _pd_0: TimeLikeArray[int] = ARRAYS1D.PD_NP.INT
        _pd_1: TimeLikeArray[int] = ARRAYS1D.PD_PA.INT
        _pl_0: TimeLikeArray[int] = ARRAYS1D.PL.INT
        # fmt: on

    # datetime arrays
    def test_datetime(self) -> None:
        # fmt: off
        _np_0: TimeLikeArray[dt.datetime, dt.timedelta] = ARRAYS1D.NP.DATETIME
        _pd_0: TimeLikeArray[dt.datetime, dt.timedelta] = ARRAYS1D.PD_NP.DATETIME
        _pd_1: TimeLikeArray[dt.datetime, dt.timedelta] = ARRAYS1D.PD_PA.DATETIME
        _pl_0: TimeLikeArray[dt.datetime, dt.timedelta] = ARRAYS1D.PL.DATETIME
        # fmt: on


class TestSpanLikeArrayInference:
    r"""Static type tests for SpanLikeArray assignments."""

    def test_float(self) -> None:
        # fmt: off
        _np_0: SpanLikeArray[float] = ARRAYS1D.NP.FLOAT
        _pd_0: SpanLikeArray[float] = ARRAYS1D.PD_NP.FLOAT
        _pd_1: SpanLikeArray[float] = ARRAYS1D.PD_PA.FLOAT
        _pl_0: SpanLikeArray[float] = ARRAYS1D.PL.FLOAT
        # fmt: on

    def test_int(self) -> None:
        # fmt: off
        _np_0: SpanLikeArray[int] = ARRAYS1D.NP.INT
        _pd_0: SpanLikeArray[int] = ARRAYS1D.PD_NP.INT
        _pd_1: SpanLikeArray[int] = ARRAYS1D.PD_PA.INT
        _pl_0: SpanLikeArray[int] = ARRAYS1D.PL.INT
        # fmt: on

    def test_timedelta(self) -> None:
        # fmt: off
        _np_0: SpanLikeArray[dt.timedelta] = ARRAYS1D.NP.TIMEDELTA
        _pd_0: SpanLikeArray[dt.timedelta] = ARRAYS1D.PD_NP.TIMEDELTA
        _pd_1: SpanLikeArray[dt.timedelta] = ARRAYS1D.PD_PA.TIMEDELTA
        _pl_0: SpanLikeArray[dt.timedelta] = ARRAYS1D.PL.TIMEDELTA
        # fmt: on
