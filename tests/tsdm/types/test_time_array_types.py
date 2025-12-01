r"""Test the timestamp protocol on arrays."""

from datetime import datetime as py_datetime, timedelta as py_timedelta
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.typing import NDArray

from tsdm.backend.types import DurationArray, TimestampArray
from tsdm.types.scalars import DurationScalar, TimestampScalar
from tsdm.utils import timedelta, timestamp

# region setup -------------------------------------------------------------------------
type np_int = np.int64  # noqa: PYI042
type np_float = np.float64  # noqa: PYI042
type np_timedelta = "np.timedelta64[py_timedelta]"  # noqa: PYI042
type np_datetime = "np.datetime64[py_datetime]"  # noqa: PYI042
type pd_datetime = pd.Timestamp  # noqa: PYI042
type pd_timedelta = pd.Timedelta  # noqa: PYI042
# fmt: off
ISO_DATE = "2021-01-01"
PY_FLOAT     : float        = float(10)
PY_INT       : int          = int(10.0)
PY_DATETIME  : py_datetime  = py_datetime.fromisoformat(ISO_DATE)
PY_TIMEDELTA : py_timedelta = py_timedelta(days=1)
NP_FLOAT     : np_float     = np.float64(10.0)
NP_INT       : np_int       = np.int64(10)
NP_TIMEDELTA : np_timedelta = np.timedelta64(1, "D")
NP_DATETIME  : np_datetime  = np.datetime64(ISO_DATE)
PD_DATETIME  : pd_datetime  = timestamp(ISO_DATE)
PD_TIMEDELTA : pd_timedelta = timedelta(days=1)
# fmt: on
# region array datetimes ---------------------------------------------------------------
TS_NUMPY_DATE: NDArray[np.datetime64] = np.array([NP_DATETIME])
TS_NUMPY_FLOAT: NDArray[np.float64] = np.array([NP_FLOAT])
TS_NUMPY_INT: NDArray[np.int64] = np.array([NP_INT])
TS_PANDAS_NP_DATE: pd.Series = pd.Series([PD_DATETIME])
TS_PANDAS_NP_FLOAT: pd.Series = pd.Series([PY_FLOAT], dtype="float64")
TS_PANDAS_NP_INT: pd.Series = pd.Series([PY_INT], dtype="int64")
TS_PANDAS_PA_DATE: pd.Series = pd.Series([PD_DATETIME], dtype="timestamp[ms][pyarrow]")
TS_PANDAS_PA_FLOAT: pd.Series = pd.Series([PY_FLOAT], dtype="float64[pyarrow]")
TS_PANDAS_PA_INT: pd.Series = pd.Series([PY_INT], dtype="int64[pyarrow]")
TS_POLARS_DATE: pl.Series = pl.Series([PD_DATETIME], dtype=pl.Date())
TS_POLARS_FLOAT: pl.Series = pl.Series([PY_FLOAT], dtype=pl.Float64())
TS_POLARS_INT: pl.Series = pl.Series([PY_INT], dtype=pl.Int64())
# endregion array datetimes ------------------------------------------------------------

# region array timedeltas --------------------------------------------------------------
TD_NUMPY_DUR: NDArray[np.timedelta64] = np.array(
    [PY_TIMEDELTA], dtype="timedelta64[ns]"
)
TD_NUMPY_FLOAT: NDArray[np.float64] = np.array([PY_FLOAT], dtype="float64")
TD_NUMPY_INT: NDArray[np.int64] = np.array([PY_INT], dtype="int64")
TD_PANDAS_NP_DUR: pd.Series = pd.Series([PY_TIMEDELTA], dtype="timedelta64[ns]")
TD_PANDAS_NP_FLOAT: pd.Series = pd.Series([PY_FLOAT], dtype="float64")
TD_PANDAS_NP_INT: pd.Series = pd.Series([PY_INT], dtype="int64")
TD_PANDAS_PA_DUR: pd.Series = pd.Series([PY_TIMEDELTA], dtype="duration[ns][pyarrow]")
TD_PANDAS_PA_FLOAT: pd.Series = pd.Series([PY_FLOAT], dtype="float64[pyarrow]")
TD_PANDAS_PA_INT: pd.Series = pd.Series([PY_INT], dtype="int64[pyarrow]")
TD_POLARS_DUR: pl.Series = pl.Series([PY_TIMEDELTA], dtype=pl.Duration())
TD_POLARS_FLOAT: pl.Series = pl.Series([PY_FLOAT], dtype=pl.Float64())
TD_POLARS_INT: pl.Series = pl.Series([PY_INT], dtype=pl.Int64())
# endregion array timedeltas -----------------------------------------------------------

type KEY_NP = Literal["numpy[float]", "numpy[int]", "numpy[time]"]
type KEY_PD = Literal["pandas[np_float]", "pandas[np_int]", "pandas[np_time]"]
type KEY_PA = Literal["pandas[pa_float]", "pandas[pa_int]", "pandas[pa_time]"]
type KEY_PL = Literal["polars[float]", "polars[int]", "polars[time]"]
type KEY = str  # KEY_NP | KEY_PD | KEY_PA | KEY_PL
TEST_CASES: list[KEY] = [
    "numpy[float]", "numpy[int]", "numpy[time]",
    "pandas[np_time]", "pandas[np_float]", "pandas[np_int]",
    "pandas[pa_time]", "pandas[pa_float]", "pandas[pa_int]",
    "polars[time]", "polars[float]", "polars[int]",
]  # fmt: skip
# endregion setup ----------------------------------------------------------------------

# region test data ---------------------------------------------------------------------
TIMEDELTA_ARRAYS: dict[KEY, DurationArray] = {
    "numpy[float]"     : TD_NUMPY_FLOAT,
    "numpy[int]"       : TD_NUMPY_INT,
    "numpy[time]"      : TD_NUMPY_DUR,
    "pandas[np_float]" : TD_PANDAS_NP_FLOAT,
    "pandas[np_int]"   : TD_PANDAS_NP_INT,
    "pandas[np_time]"  : TD_PANDAS_NP_DUR,
    "pandas[pa_float]" : TD_PANDAS_PA_FLOAT,
    "pandas[pa_int]"   : TD_PANDAS_PA_INT,
    "pandas[pa_time]"  : TD_PANDAS_PA_DUR,
    "polars[float]"    : TD_POLARS_FLOAT,
    "polars[int]"      : TD_POLARS_INT,
    "polars[time]"     : TD_POLARS_DUR,
}  # fmt: skip
r"""Dictionary of timedelta arrays."""

TIMEDELTA_SCALARS: dict[KEY, DurationScalar] = {
    "numpy[float]"     : NP_FLOAT,
    "numpy[int]"       : NP_INT,
    "numpy[time]"      : NP_TIMEDELTA,
    "pandas[np_time]"  : PY_TIMEDELTA,
    "pandas[np_float]" : PY_FLOAT,
    "pandas[np_int]"   : PY_INT,
    "pandas[pa_time]"  : PY_TIMEDELTA,
    "pandas[pa_float]" : PY_FLOAT,
    "pandas[pa_int]"   : PY_INT,
    "polars[time]"     : PY_TIMEDELTA,
    "polars[float]"    : PY_FLOAT,
    "polars[int]"      : PY_INT,
}  # fmt: skip
r"""Dictionary of compatible python timedelta values for each timedelta."""

TIMESTAMP_ARRAYS: dict[KEY, TimestampArray] = {
    "numpy[float]"     : TS_NUMPY_FLOAT,
    "numpy[int]"       : TS_NUMPY_INT,
    "numpy[time]"      : TS_NUMPY_DATE,
    "pandas[np_time]"  : TS_PANDAS_NP_DATE,
    "pandas[np_float]" : TS_PANDAS_NP_FLOAT,
    "pandas[np_int]"   : TS_PANDAS_NP_INT,
    "pandas[pa_time]"  : TS_PANDAS_PA_DATE,
    "pandas[pa_float]" : TS_PANDAS_PA_FLOAT,
    "pandas[pa_int]"   : TS_PANDAS_PA_INT,
    "polars[time]"     : TS_POLARS_DATE,
    "polars[float]"    : TS_POLARS_FLOAT,
    "polars[int]"      : TS_POLARS_INT,
}  # fmt: skip
r"""Dictionary of timestamp arrays."""

TIMESTAMP_SCALARS: dict[KEY, TimestampScalar] = {
    "numpy[float]"     : NP_FLOAT,
    "numpy[int]"       : NP_INT,
    "numpy[time]"      : NP_DATETIME,
    "pandas[np_time]"  : PY_DATETIME,
    "pandas[np_float]" : PY_FLOAT,
    "pandas[np_int]"   : PY_INT,
    "pandas[pa_time]"  : PY_DATETIME,
    "pandas[pa_float]" : PY_FLOAT,
    "pandas[pa_int]"   : PY_INT,
    "polars[time]"     : PY_DATETIME,
    "polars[float]"    : PY_FLOAT,
    "polars[int]"      : PY_INT,
}  # fmt: skip
r"""Dictionary of compatible python datetime values for each datetime."""
# endregion test data ------------------------------------------------------------------


class TestDurationArrayProtocol:
    r"""Test the duration array protocol."""

    # float arrays
    _numpy_float: DurationArray[float] = TD_NUMPY_FLOAT
    _pandas_np_float: DurationArray[float] = TD_PANDAS_NP_FLOAT
    _pandas_pa_float: DurationArray[float] = TD_PANDAS_PA_FLOAT
    _polars_float: DurationArray[float] = TD_POLARS_FLOAT
    # int arrays
    _numpy_int: DurationArray[int] = TD_NUMPY_INT
    _pandas_np_int: DurationArray[int] = TD_PANDAS_NP_INT
    _pandas_pa_int: DurationArray[int] = TD_PANDAS_PA_INT
    _polars_int: DurationArray[int] = TD_POLARS_INT
    # timedelta arrays
    # _numpy_time: DurationArray[py_timedelta] = TD_NUMPY_DUR
    _pandas_np_time: DurationArray[py_timedelta] = TD_PANDAS_NP_DUR
    _pandas_pa_time: DurationArray[py_timedelta] = TD_PANDAS_PA_DUR
    _polars_time: DurationArray[py_timedelta] = TD_POLARS_DUR

    DURATION_FLOAT_ARRAYS: dict[KEY, DurationArray[float]] = {
        "numpy[float]"     : TD_NUMPY_FLOAT,
        "pandas[np_float]" : TD_PANDAS_NP_FLOAT,
        "pandas[pa_float]" : TD_PANDAS_PA_FLOAT,
        "polars[float]"    : TD_POLARS_FLOAT,
    }  # fmt: skip
    r"""Dictionary of float arrays."""

    DURATION_INT_ARRAYS: dict[KEY, DurationArray[int]] = {
        "numpy[int]": TD_NUMPY_INT,
        "pandas[np_int]": TD_PANDAS_NP_INT,
        "pandas[pa_int]": TD_PANDAS_PA_INT,
        "polars[int]": TD_POLARS_INT,
    }  # fmt: skip
    r"""Dictionary of int arrays."""

    DURATION_TIMEDELTA_ARRAYS: dict[KEY, DurationArray[py_timedelta]] = {
        # "numpy[time]"     : TD_NUMPY_DUR,
        "pandas[np_time]": TD_PANDAS_NP_DUR,
        "pandas[pa_time]": TD_PANDAS_PA_DUR,
        "polars[time]": TD_POLARS_DUR,
    }  # fmt: skip
    r"""Dictionary of timedelta arrays."""

    @pytest.mark.parametrize("case", TEST_CASES)
    def test_timedelta_arrays(self, case) -> None:
        td_array = TIMEDELTA_ARRAYS[case]
        td_scalar = TIMEDELTA_SCALARS[case]
        ts_scalar = TIMESTAMP_SCALARS[case]
        cls = type(td_array)

        # unary operations
        assert type(+td_array) is cls
        assert type(-td_array) is cls
        assert type(abs(td_array)) is cls
        # arithmetic
        assert type(td_array + td_scalar) is cls
        assert type(td_array - td_scalar) is cls
        assert type(ts_scalar + td_array) is cls
        assert type(ts_scalar - td_array) is cls
        # comparisons
        assert type(td_array < td_scalar) is cls
        # multiplication with int
        assert type(td_array * PY_INT) is cls
        assert type(PY_INT * td_array) is cls
        # floor division with int
        assert type(td_array // PY_INT) is cls

    @pytest.mark.parametrize("case", DURATION_FLOAT_ARRAYS)
    def test_duration_float_arrays(self, case: str) -> None:
        td_array = self.DURATION_FLOAT_ARRAYS[case]
        cls = type(td_array)

        # addition
        assert type(PY_FLOAT + td_array) is cls
        assert type(td_array + PY_FLOAT) is cls
        # subtraction
        assert type(td_array - PY_FLOAT) is cls
        assert type(PY_FLOAT - td_array) is cls
        # comparisons
        assert type(td_array < PY_FLOAT) is cls

    @pytest.mark.parametrize("case", DURATION_INT_ARRAYS)
    def test_duration_int_arrays(self, case: str) -> None:
        td_array = self.DURATION_INT_ARRAYS[case]
        cls = type(td_array)

        # addition
        assert type(td_array + PY_INT) is cls
        assert type(PY_INT + td_array) is cls
        # subtraction
        assert type(td_array - PY_INT) is cls
        assert type(PY_INT - td_array) is cls
        # comparisons
        assert type(td_array < PY_INT) is cls

    @pytest.mark.parametrize("case", DURATION_TIMEDELTA_ARRAYS)
    def test_duration_timedelta_arrays(self, case: str) -> None:
        td_array = self.DURATION_TIMEDELTA_ARRAYS[case]
        cls = type(td_array)

        # addition
        assert type(td_array + PY_TIMEDELTA) is cls
        assert type(PY_TIMEDELTA + td_array) is cls
        # subtraction
        assert type(td_array - PY_TIMEDELTA) is cls
        assert type(PY_TIMEDELTA - td_array) is cls
        # comparisons
        assert type(td_array < PY_TIMEDELTA) is cls


class TestTimestampArrayProtocol:
    r"""Test the timestamp array protocol."""

    # float arrays
    _numpy_float: TimestampArray[TimestampScalar[float]] = TS_NUMPY_FLOAT
    _pandas_np_float: TimestampArray[TimestampScalar[float]] = TS_PANDAS_NP_FLOAT
    _pandas_pa_float: TimestampArray[TimestampScalar[float]] = TS_PANDAS_PA_FLOAT
    _polars_float: TimestampArray[TimestampScalar[float]] = TS_POLARS_FLOAT

    # int arrays
    _numpy_int: TimestampArray[TimestampScalar[int]] = TS_NUMPY_INT
    _pandas_np_int: TimestampArray[TimestampScalar[int]] = TS_PANDAS_NP_INT
    _pandas_pa_int: TimestampArray[TimestampScalar[int]] = TS_PANDAS_PA_INT
    _polars_int: TimestampArray[TimestampScalar[int]] = TS_POLARS_INT

    # datetime arrays
    # _numpy_time: TimestampArray[TimestampScalar[py_timedelta]] = TS_NUMPY_DATE
    _pandas_np_time: TimestampArray[TimestampScalar[py_timedelta]] = TS_PANDAS_NP_DATE
    _pandas_pa_time: TimestampArray[TimestampScalar[py_timedelta]] = TS_PANDAS_PA_DATE
    _polars_time: TimestampArray[TimestampScalar[py_timedelta]] = TS_POLARS_DATE

    TIMESTAMP_FLOAT_ARRAYS: dict[KEY, TimestampArray[TimestampScalar[float]]] = {
        "numpy[float]": TS_NUMPY_FLOAT,
        "pandas[np_float]": TS_PANDAS_NP_FLOAT,
        "pandas[pa_float]": TS_PANDAS_PA_FLOAT,
        "polars[float]": TS_POLARS_FLOAT,
    }  # fmt: skip
    r"""Dictionary of float arrays."""
    TIMESTAMP_INT_ARRAYS: dict[KEY, TimestampArray[TimestampScalar[int]]] = {
        "numpy[int]": TS_NUMPY_INT,
        "pandas[np_int]": TS_PANDAS_NP_INT,
        "pandas[pa_int]": TS_PANDAS_PA_INT,
        "polars[int]": TS_POLARS_INT,
    }  # fmt: skip
    r"""Dictionary of int arrays."""
    TIMESTAMP_PYDATETIME_ARRAYS: dict[KEY, TimestampArray[TimestampScalar[py_timedelta]]] = {
        # "numpy[time]"     : TS_NUMPY_DATE,
        "pandas[np_time]": TS_PANDAS_NP_DATE,
        "pandas[pa_time]": TS_PANDAS_PA_DATE,
        "polars[time]": TS_POLARS_DATE,
    }  # fmt: skip
    r"""Dictionary of datetime arrays."""

    @pytest.mark.parametrize("case", TEST_CASES)
    def test_timestamp_arrays(self, case: KEY) -> None:
        ts_array = TIMESTAMP_ARRAYS[case]
        td_scalar = TIMEDELTA_SCALARS[case]
        ts_scalar = TIMESTAMP_SCALARS[case]
        cls = type(ts_array)

        # comparisons
        assert type(ts_array < ts_scalar) is cls
        # arithmetic
        assert type(ts_array + td_scalar) is cls
        assert type(ts_array - td_scalar) is cls

    @pytest.mark.parametrize("case", TIMESTAMP_FLOAT_ARRAYS)
    def test_timestamp_float_arrays(self, case: str) -> None:
        ts_array = self.TIMESTAMP_FLOAT_ARRAYS[case]
        cls = type(ts_array)

        # arithmetic
        assert type(ts_array + PY_FLOAT) is cls
        assert type(PY_FLOAT + ts_array) is cls
        assert type(ts_array - PY_FLOAT) is cls
        assert type(PY_FLOAT - ts_array) is cls
        # comparisons
        assert type(ts_array < PY_FLOAT) is cls

    @pytest.mark.parametrize("case", TIMESTAMP_INT_ARRAYS)
    def test_timestamp_int_arrays(self, case: str) -> None:
        ts_array = self.TIMESTAMP_INT_ARRAYS[case]
        cls = type(ts_array)

        # arithmetic
        assert type(ts_array + PY_INT) is cls
        assert type(PY_INT + ts_array) is cls
        assert type(ts_array - PY_INT) is cls
        assert type(PY_INT - ts_array) is cls
        # comparisons
        assert type(ts_array < PY_INT) is cls

    @pytest.mark.parametrize("case", TIMESTAMP_PYDATETIME_ARRAYS)
    def test_timestamp_datetime_arrays(self, case: str) -> None:
        ts_array = self.TIMESTAMP_PYDATETIME_ARRAYS[case]
        cls = type(ts_array)

        # addition
        assert type(ts_array + PY_TIMEDELTA) is cls
        assert type(PY_TIMEDELTA + ts_array) is cls
        # subtraction
        assert type(ts_array - PY_DATETIME) is cls
        assert type(PY_DATETIME - ts_array) is cls
        # comparisons
        assert type(ts_array < PY_DATETIME) is cls
