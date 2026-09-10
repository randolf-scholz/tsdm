r"""Test the timestamp protocol on arrays."""

import datetime as dt
from typing import TYPE_CHECKING, Literal

import pytest

from experimental.numerical_types import (
    DatetimeArray,
    SpanLikeArray,
    SpanLikeScalar,
    TimedeltaArray,
    TimeLikeArray,
    TimeLikeScalar,
)
from test_utils import pytest_xfail
from experimental.tests.fixtures import ARRAYS1D, SCALARS

# region setup -------------------------------------------------------------------------
type KEY_NP = Literal["numpy[np_float]", "numpy[np_int]", "numpy[np_time]"]
type KEY_PD = Literal["pandas[np_float]", "pandas[np_int]", "pandas[np_time]"]
type KEY_PA = Literal["pandas[pa_float]", "pandas[pa_int]", "pandas[pa_time]"]
type KEY_PL = Literal["polars[pa_float]", "polars[pa_int]", "polars[pa_time]"]
type KEY = str  # | KEY_NP | KEY_PD | KEY_PA | KEY_PL
TEST_CASES: list[KEY] = [
    "numpy[np_float]", "numpy[np_int]", "numpy[np_time]",
    "pandas[np_time]", "pandas[np_float]", "pandas[np_int]",
    "pandas[pa_time]", "pandas[pa_float]", "pandas[pa_int]",
    "polars[pa_time]", "polars[pa_float]", "polars[pa_int]",
]  # fmt: skip
# endregion setup ----------------------------------------------------------------------

# region test data ---------------------------------------------------------------------
DURATION_ARRAYS: dict[KEY, SpanLikeArray] = {
    "numpy[np_float]"  : ARRAYS1D.NP.FLOAT,
    "numpy[np_int]"    : ARRAYS1D.NP.INT,
    "numpy[np_time]"   : ARRAYS1D.NP.TIMEDELTA,
    "pandas[np_float]" : ARRAYS1D.PD_NP.FLOAT,
    "pandas[np_int]"   : ARRAYS1D.PD_NP.INT,
    "pandas[np_time]"  : ARRAYS1D.PD_NP.TIMEDELTA,
    "pandas[pa_float]" : ARRAYS1D.PD_PA.FLOAT,
    "pandas[pa_int]"   : ARRAYS1D.PD_PA.INT,
    "pandas[pa_time]"  : ARRAYS1D.PD_PA.TIMEDELTA,
    "polars[pa_float]" : ARRAYS1D.PL.FLOAT,
    "polars[pa_int]"   : ARRAYS1D.PL.INT,
    "polars[pa_time]"  : ARRAYS1D.PL.TIMEDELTA,
    "torch[float]"     : ARRAYS1D.PT.FLOAT,
    "torch[int]"       : ARRAYS1D.PT.INT,
}  # fmt: skip
r"""Dictionary of timedelta arrays."""

DURATION_SCALARS: dict[KEY, SpanLikeScalar] = {
    "numpy[np_float]"  : SCALARS.NP.FLOAT,
    "numpy[np_int]"    : SCALARS.NP.INT,
    "numpy[np_time]"   : SCALARS.NP.TIMEDELTA,
    "pandas[np_time]"  : SCALARS.PD.TIMEDELTA,
    "pandas[np_float]" : SCALARS.PY.FLOAT,
    "pandas[np_int]"   : SCALARS.PY.INT,
    "pandas[pa_time]"  : SCALARS.PD.TIMEDELTA,
    "pandas[pa_float]" : SCALARS.PY.FLOAT,
    "pandas[pa_int]"   : SCALARS.PY.INT,
    "polars[pa_time]"  : SCALARS.PY.TIMEDELTA,
    "polars[pa_float]" : SCALARS.PY.FLOAT,
    "polars[pa_int]"   : SCALARS.PY.INT,
    "torch[float]"     : SCALARS.PY.FLOAT,
    "torch[int]"       : SCALARS.PY.INT,
}  # fmt: skip
r"""Dictionary of compatible python timedelta values for each timedelta."""

TIMEDELTA_ARRAYS: dict[KEY, TimedeltaArray] = {
    "numpy[np_time]"  : ARRAYS1D.NP.TIMEDELTA,
    "pandas[np_time]" : ARRAYS1D.PD_NP.TIMEDELTA,
    "pandas[pa_time]" : ARRAYS1D.PD_PA.TIMEDELTA,
    "polars[pa_time]" : ARRAYS1D.PL.TIMEDELTA,
}  # fmt: skip
r"""Dictionary of timedelta arrays."""

TIMEDELTA_SCALARS: dict[KEY, TimedeltaArray] = {
    "numpy[np_time]"  : SCALARS.NP.TIMEDELTA,
    "pandas[np_time]" : SCALARS.PD.TIMEDELTA,
    "pandas[pa_time]" : SCALARS.PD.TIMEDELTA,
    "polars[pa_time]" : SCALARS.PY.TIMEDELTA,
}  # fmt: skip
r"""Dictionary of compatible python timedelta values for each timedelta."""

TIMESTAMP_ARRAYS: dict[KEY, TimeLikeArray] = {
    "numpy[np_float]"  : ARRAYS1D.NP.FLOAT,
    "numpy[np_int]"    : ARRAYS1D.NP.INT,
    "numpy[np_time]"   : ARRAYS1D.NP.DATETIME,
    "pandas[np_time]"  : ARRAYS1D.PD_NP.DATETIME,
    "pandas[np_float]" : ARRAYS1D.PD_NP.FLOAT,
    "pandas[np_int]"   : ARRAYS1D.PD_NP.INT,
    "pandas[pa_time]"  : ARRAYS1D.PD_PA.DATETIME,
    "pandas[pa_float]" : ARRAYS1D.PD_PA.FLOAT,
    "pandas[pa_int]"   : ARRAYS1D.PD_PA.INT,
    "polars[pa_time]"  : ARRAYS1D.PL.DATETIME,
    "polars[pa_float]" : ARRAYS1D.PL.FLOAT,
    "polars[pa_int]"   : ARRAYS1D.PL.INT,
}  # fmt: skip
r"""Dictionary of timestamp arrays."""

DATETIME_ARRAYS: dict[KEY, DatetimeArray] = {
    "numpy[np_time]"   : ARRAYS1D.NP.DATETIME,  # pyright: ignore[reportAssignmentType]
    "pandas[np_time]"  : ARRAYS1D.PD_NP.DATETIME,
    "pandas[pa_time]"  : ARRAYS1D.PD_PA.DATETIME,
    "polars[pa_time]"  : ARRAYS1D.PL.DATETIME,
}  # fmt: skip
r"""Dictionary of datetime arrays."""

TIMESTAMP_SCALARS: dict[KEY, TimeLikeScalar] = {
    "numpy[np_float]"  : SCALARS.NP.FLOAT,
    "numpy[np_int]"    : SCALARS.NP.INT,
    "numpy[np_time]"   : SCALARS.NP.DATETIME,
    "pandas[np_time]"  : SCALARS.PY.DATETIME,
    "pandas[np_float]" : SCALARS.PY.FLOAT,
    "pandas[np_int]"   : SCALARS.PY.INT,
    "pandas[pa_time]"  : SCALARS.PY.DATETIME,
    "pandas[pa_float]" : SCALARS.PY.FLOAT,
    "pandas[pa_int]"   : SCALARS.PY.INT,
    "polars[pa_time]"  : SCALARS.PY.DATETIME,
    "polars[pa_float]" : SCALARS.PY.FLOAT,
    "polars[pa_int]"   : SCALARS.PY.INT,
}  # fmt: skip
r"""Dictionary of compatible python datetime values for each datetime."""
# endregion test data ------------------------------------------------------------------


class TestDurationArrayProtocol:
    r"""Test the duration array protocol."""

    DURATION_INT_ARRAYS: dict[KEY, SpanLikeArray[int]] = {
        "numpy[np_int]"  : ARRAYS1D.INT.NP,
        "pandas[np_int]" : ARRAYS1D.INT.PD_NP,
        "pandas[pa_int]" : ARRAYS1D.INT.PD_PA,
        "polars[pa_int]" : ARRAYS1D.INT.PL,
        "torch[int]"     : ARRAYS1D.INT.PT,
    }  # fmt: skip
    r"""Dictionary of int arrays."""

    DURATION_FLOAT_ARRAYS: dict[KEY, SpanLikeArray[float]] = {
        "numpy[np_float]"  : ARRAYS1D.FLOAT.NP,
        "pandas[np_float]" : ARRAYS1D.FLOAT.PD_NP,
        "pandas[pa_float]" : ARRAYS1D.FLOAT.PD_PA,
        "polars[pa_float]" : ARRAYS1D.FLOAT.PL,
        "torch[float]"     : ARRAYS1D.FLOAT.PT,
    }  # fmt: skip
    r"""Dictionary of float arrays."""

    DURATION_TIMEDELTA_ARRAYS: dict[KEY, SpanLikeArray[dt.timedelta]] = {
        "numpy[np_time]" : ARRAYS1D.TIMEDELTA.NP,
        "pandas[np_time]": ARRAYS1D.TIMEDELTA.PD_NP,
        "pandas[pa_time]": ARRAYS1D.TIMEDELTA.PD_PA,
        "polars[pa_time]": ARRAYS1D.TIMEDELTA.PL,
    }  # fmt: skip
    r"""Dictionary of timedelta arrays."""

    @pytest.mark.parametrize("case", TEST_CASES)
    def test_duration_arrays(self, case: str) -> None:
        # covers int, float and timedelta cases
        td_array = DURATION_ARRAYS[case]
        td_scalar = DURATION_SCALARS[case]
        ts_scalar = TIMESTAMP_SCALARS[case]
        cls = type(td_array)

        # comparisons
        assert type(td_array < td_array) is cls
        assert type(td_array < td_scalar) is cls
        assert type(td_array == td_array) is cls
        assert type(td_array == td_scalar) is cls
        # unary operations
        assert type(+td_array) is cls
        assert type(-td_array) is cls
        assert type(abs(td_array)) is cls
        # arithmetic
        assert type(td_array + td_scalar) is cls
        assert type(td_array - td_scalar) is cls
        assert type(ts_scalar + td_array) is cls
        assert type(ts_scalar - td_array) is cls
        # multiplication with int
        assert type(td_array * SCALARS.PY.INT) is cls
        assert type(SCALARS.PY.INT * td_array) is cls
        # truediv with another duration|int
        assert type(td_array / td_array) is cls
        assert type(td_array / td_scalar) is cls

        if not TYPE_CHECKING:
            # FIXME: not implemented on arrow/polars
            with pytest_xfail(condition=case == "polars[pa_time]", defer_xfail=True):
                assert type(td_array // td_array) is cls
            with pytest_xfail(condition=case == "polars[pa_time]", defer_xfail=True):
                assert type(td_array // td_scalar) is cls
            with pytest_xfail(condition="pa_time" in case, defer_xfail=True):
                assert type(td_array // SCALARS.PY.INT) is cls
            with pytest_xfail(
                condition="pandas[pa_" in case or "pa_time" in case,
                defer_xfail=True,
            ):
                assert type(td_array % td_array) is cls
            with pytest_xfail(
                condition="pandas[pa_" in case or "pa_time" in case,
                defer_xfail=True,
            ):
                assert type(td_array % td_scalar) is cls

    @pytest.mark.parametrize("case", TIMEDELTA_ARRAYS)
    def test_timedelta_arrays(self, case: str) -> None:
        td_array = TIMEDELTA_ARRAYS[case]
        cls = type(td_array)

        assert type(td_array < SCALARS.PY.TIMEDELTA) is cls
        assert type(td_array == SCALARS.PY.TIMEDELTA) is cls
        assert type(td_array / SCALARS.PY.TIMEDELTA) is cls

        if not TYPE_CHECKING:
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(td_array + SCALARS.PY.DATETIME) is cls
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(SCALARS.PY.DATETIME + td_array) is cls
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(td_array + SCALARS.PY.TIMEDELTA) is cls
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(SCALARS.PY.TIMEDELTA + td_array) is cls
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(td_array - SCALARS.PY.TIMEDELTA) is cls
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(SCALARS.PY.TIMEDELTA - td_array) is cls
            with pytest_xfail(condition=case == "polars[pa_time]", defer_xfail=True):
                assert type(td_array // SCALARS.PY.TIMEDELTA) is cls
            with pytest_xfail(condition=case != "pandas[np_time]", defer_xfail=True):
                assert type(td_array % SCALARS.PY.TIMEDELTA) is cls

    @pytest.mark.parametrize("case", DURATION_FLOAT_ARRAYS)
    def test_duration_float_arrays(self, case: str) -> None:
        td_array = self.DURATION_FLOAT_ARRAYS[case]
        cls = type(td_array)

        # addition
        assert type(SCALARS.PY.FLOAT + td_array) is cls
        assert type(td_array + SCALARS.PY.FLOAT) is cls
        # subtraction
        assert type(td_array - SCALARS.PY.FLOAT) is cls
        assert type(SCALARS.PY.FLOAT - td_array) is cls
        # comparisons
        assert type(td_array < SCALARS.PY.FLOAT) is cls

    @pytest.mark.parametrize("case", DURATION_INT_ARRAYS)
    def test_duration_int_arrays(self, case: str) -> None:
        td_array = self.DURATION_INT_ARRAYS[case]
        cls = type(td_array)

        # addition
        assert type(td_array + SCALARS.PY.INT) is cls
        assert type(SCALARS.PY.INT + td_array) is cls
        # subtraction
        assert type(td_array - SCALARS.PY.INT) is cls
        assert type(SCALARS.PY.INT - td_array) is cls
        # comparisons
        assert type(td_array < SCALARS.PY.INT) is cls

    @pytest_xfail(
        condition=lambda _, case: case == "numpy[np_time]",
        reason="numpy does not support addition of timedelta arrays with datetime scalars",
        strict=True,
    )
    @pytest.mark.parametrize("case", DURATION_TIMEDELTA_ARRAYS)
    def test_duration_timedelta_arrays(self, case: str) -> None:
        td_array = self.DURATION_TIMEDELTA_ARRAYS[case]
        cls = type(td_array)

        # addition
        assert type(td_array + SCALARS.PY.TIMEDELTA) is cls
        assert type(SCALARS.PY.TIMEDELTA + td_array) is cls
        # subtraction
        assert type(td_array - SCALARS.PY.TIMEDELTA) is cls
        assert type(SCALARS.PY.TIMEDELTA - td_array) is cls
        # comparisons
        assert type(td_array < SCALARS.PY.TIMEDELTA) is cls


class TestTimestampArrayProtocol:
    r"""Test the timestamp array protocol."""

    TIMESTAMP_FLOAT_ARRAYS: dict[KEY, TimeLikeArray[float]] = {
        "numpy[np_float]" : ARRAYS1D.FLOAT.NP,
        "pandas[np_float]": ARRAYS1D.FLOAT.PD_NP,
        "pandas[pa_float]": ARRAYS1D.FLOAT.PD_PA,
        "polars[pa_float]": ARRAYS1D.FLOAT.PL,
        "torch[float]"    : ARRAYS1D.FLOAT.PT,
    }  # fmt: skip
    r"""Dictionary of float arrays."""
    TIMESTAMP_INT_ARRAYS: dict[KEY, TimeLikeArray[int]] = {
        "numpy[np_int]" : ARRAYS1D.INT.NP,
        "pandas[np_int]": ARRAYS1D.INT.PD_NP,
        "pandas[pa_int]": ARRAYS1D.INT.PD_PA,
        "polars[pa_int]": ARRAYS1D.INT.PL,
        "torch[int]"    : ARRAYS1D.INT.PT,
    }  # fmt: skip
    r"""Dictionary of int arrays."""
    TIMESTAMP_PYDATETIME_ARRAYS: dict[KEY, TimeLikeArray[dt.datetime]] = {
        "numpy[np_time]" : ARRAYS1D.DATETIME.NP,
        "pandas[np_time]": ARRAYS1D.DATETIME.PD_NP,
        "pandas[pa_time]": ARRAYS1D.DATETIME.PD_PA,
        "polars[pa_time]": ARRAYS1D.DATETIME.PL,
    }  # fmt: skip
    r"""Dictionary of datetime arrays."""

    @pytest.mark.parametrize("case", TIMESTAMP_ARRAYS)
    def test_timestamp_arrays(self, case: KEY) -> None:
        ts_array = TIMESTAMP_ARRAYS[case]
        td_scalar = DURATION_SCALARS[case]
        td_array = DURATION_ARRAYS[case]
        ts_scalar = TIMESTAMP_SCALARS[case]
        cls = type(ts_array)

        # comparisons
        assert type(ts_array < ts_scalar) is cls
        assert type(ts_array < ts_array) is cls
        assert type(ts_array == ts_scalar) is cls
        assert type(ts_array == ts_array) is cls

        # arithmetic
        assert type(ts_array + td_scalar) is cls
        assert type(td_scalar + ts_array) is cls
        assert type(ts_array + td_array) is cls
        assert type(td_array + ts_array) is cls
        assert type(ts_array - td_scalar) is cls
        assert type(ts_array - td_array) is cls

    @pytest.mark.parametrize("case", DATETIME_ARRAYS)
    def test_datetime_arrays(self, case: KEY) -> None:
        ts_array = DATETIME_ARRAYS[case]
        cls = type(ts_array)

        # comparisons
        assert type(ts_array < SCALARS.PY.DATETIME) is cls
        assert type(ts_array == SCALARS.PY.DATETIME) is cls

        # arithmetic
        if not TYPE_CHECKING:
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(ts_array + SCALARS.PY.TIMEDELTA) is cls
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(SCALARS.PY.TIMEDELTA + ts_array) is cls
            with pytest_xfail(condition=case == "numpy[np_time]", defer_xfail=True):
                assert type(ts_array - SCALARS.PY.TIMEDELTA) is cls

    @pytest_xfail(
        condition=lambda _, case: case == "numpy[np_time]",
        reason="numpy does not support addition of timedelta arrays with datetime scalars",
        strict=True,
    )
    @pytest.mark.parametrize("case", TIMESTAMP_PYDATETIME_ARRAYS)
    def test_timestamp_datetime_arrays(self, case: str) -> None:
        ts_array = self.TIMESTAMP_PYDATETIME_ARRAYS[case]
        cls = type(ts_array)

        # comparisons
        assert type(ts_array < SCALARS.PY.DATETIME) is cls
        # addition
        assert type(ts_array + SCALARS.PY.TIMEDELTA) is cls
        assert type(SCALARS.PY.TIMEDELTA + ts_array) is cls
        # subtraction
        assert type(ts_array - SCALARS.PY.DATETIME) is cls
        assert type(SCALARS.PY.DATETIME - ts_array) is cls

    @pytest.mark.parametrize("case", TIMESTAMP_FLOAT_ARRAYS)
    def test_timestamp_float_arrays(self, case: str) -> None:
        ts_array = self.TIMESTAMP_FLOAT_ARRAYS[case]
        cls = type(ts_array)

        # arithmetic
        assert type(ts_array + SCALARS.PY.FLOAT) is cls
        assert type(SCALARS.PY.FLOAT + ts_array) is cls
        assert type(ts_array - SCALARS.PY.FLOAT) is cls
        assert type(SCALARS.PY.FLOAT - ts_array) is cls
        # comparisons
        assert type(ts_array < SCALARS.PY.FLOAT) is cls

    @pytest.mark.parametrize("case", TIMESTAMP_INT_ARRAYS)
    def test_timestamp_int_arrays(self, case: str) -> None:
        ts_array = self.TIMESTAMP_INT_ARRAYS[case]
        cls = type(ts_array)

        # arithmetic
        assert type(ts_array + SCALARS.PY.INT) is cls
        assert type(SCALARS.PY.INT + ts_array) is cls
        assert type(ts_array - SCALARS.PY.INT) is cls
        assert type(SCALARS.PY.INT - ts_array) is cls
        # comparisons
        assert type(ts_array < SCALARS.PY.INT) is cls
