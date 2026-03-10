import datetime as dt

from numerical_types import SpanLikeArray, TimeLikeArray

from .fixtures import SERIES


class TestTimeLikeArrayAssignments:
    r"""Static type tests for TimeLikeArray assignments."""

    def test_float(self) -> None:
        # fmt: off
        _np_0: TimeLikeArray[float] = SERIES.NP.FLOAT
        _pd_0: TimeLikeArray[float] = SERIES.PD_NP.FLOAT
        _pd_1: TimeLikeArray[float] = SERIES.PD_PA.FLOAT
        _pl_0: TimeLikeArray[float] = SERIES.PL.FLOAT
        # fmt: on

    def test_int(self) -> None:
        # fmt: off
        _np_0: TimeLikeArray[int] = SERIES.NP.INT
        _pd_0: TimeLikeArray[int] = SERIES.PD_NP.INT
        _pd_1: TimeLikeArray[int] = SERIES.PD_PA.INT
        _pl_0: TimeLikeArray[int] = SERIES.PL.INT
        # fmt: on

    # datetime arrays
    def test_datetime(self) -> None:
        # fmt: off
        _np_0: TimeLikeArray[dt.datetime, dt.timedelta] = SERIES.NP.DATETIME
        _pd_0: TimeLikeArray[dt.datetime, dt.timedelta] = SERIES.PD_NP.DATETIME
        _pd_1: TimeLikeArray[dt.datetime, dt.timedelta] = SERIES.PD_PA.DATETIME
        _pl_0: TimeLikeArray[dt.datetime, dt.timedelta] = SERIES.PL.DATETIME
        # fmt: on


class TestSpanLikeArrayInference:
    r"""Static type tests for SpanLikeArray assignments."""

    def test_float(self) -> None:
        # fmt: off
        _np_0: SpanLikeArray[float] = SERIES.NP.FLOAT
        _pd_0: SpanLikeArray[float] = SERIES.PD_NP.FLOAT
        _pd_1: SpanLikeArray[float] = SERIES.PD_PA.FLOAT
        _pl_0: SpanLikeArray[float] = SERIES.PL.FLOAT
        # fmt: on

    def test_int(self) -> None:
        # fmt: off
        _np_0: SpanLikeArray[int] = SERIES.NP.INT
        _pd_0: SpanLikeArray[int] = SERIES.PD_NP.INT
        _pd_1: SpanLikeArray[int] = SERIES.PD_PA.INT
        _pl_0: SpanLikeArray[int] = SERIES.PL.INT
        # fmt: on

    def test_timedelta(self) -> None:
        # fmt: off
        _np_0: SpanLikeArray[dt.timedelta] = SERIES.NP.TIMEDELTA
        _pd_0: SpanLikeArray[dt.timedelta] = SERIES.PD_NP.TIMEDELTA
        _pd_1: SpanLikeArray[dt.timedelta] = SERIES.PD_PA.TIMEDELTA
        _pl_0: SpanLikeArray[dt.timedelta] = SERIES.PL.TIMEDELTA
        # fmt: on
