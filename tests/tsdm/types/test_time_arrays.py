r"""Test the timestamp protocol on arrays."""

from datetime import datetime as py_datetime, timedelta as py_timedelta
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import polars as pl
import pytest
from numpy.typing import NDArray

from tsdm.types.linalg import NumericalArray
from tsdm.types.scalars import DurationScalar, TimestampScalar
from tsdm.utils import timedelta, timestamp

# region setup -------------------------------------------------------------------------
ISO_DATE = "2021-01-01"
# python timedeltas
TD_PY_FLOAT: float = 10.0
TD_PY_INT: int = 10
TD_PY_DUR: py_timedelta = py_timedelta(days=1)
# numpy timedeltas
TD_NP_FLOAT: np.float64 = np.float64(10.0)
TD_NP_INT: np.int64 = np.int64(10)
TD_NP_DUR: np.timedelta64 = np.timedelta64(1, "D")
# pandas timedeltas
TD_PD_DUR: pd.Timedelta = timedelta(days=1)
# python timestamps
TS_PY_DATE: py_datetime = py_datetime.fromisoformat(ISO_DATE)
TS_PY_FLOAT: float = 10.0
TS_PY_INT: int = 10
# numpy timestamps
TS_NP_DATE: np.datetime64 = np.datetime64(ISO_DATE)
TS_NP_FLOAT: np.float64 = np.float64(TS_PY_FLOAT)
TS_NP_INT: np.int64 = np.int64(TS_PY_INT)
# pandas timestamps
TS_PD_DATE: pd.Timestamp = timestamp(ISO_DATE)

# region array datetimes ---------------------------------------------------------------
TS_NUMPY_DATE: NDArray[np.datetime64] = np.array([TS_NP_DATE])
TS_NUMPY_FLOAT: NDArray[np.float64] = np.array([TS_NP_FLOAT])
TS_NUMPY_INT: NDArray[np.int64] = np.array([TS_NP_INT])
TS_PANDAS_NP_DATE: pd.Series = pd.Series([TS_PD_DATE])
TS_PANDAS_NP_FLOAT: pd.Series = pd.Series([TS_PY_FLOAT], dtype="float64")
TS_PANDAS_NP_INT: pd.Series = pd.Series([TS_PY_INT], dtype="int64")
TS_PANDAS_PA_DATE: pd.Series = pd.Series([TS_PD_DATE], dtype="timestamp[ms][pyarrow]")
TS_PANDAS_PA_FLOAT: pd.Series = pd.Series([TS_PY_FLOAT], dtype="float64[pyarrow]")
TS_PANDAS_PA_INT: pd.Series = pd.Series([TS_PY_INT], dtype="int64[pyarrow]")
TS_POLARS_DATE: pl.Series = pl.Series([TS_PD_DATE], dtype=pl.Date())
TS_POLARS_FLOAT: pl.Series = pl.Series([TS_PY_FLOAT], dtype=pl.Float64())
TS_POLARS_INT: pl.Series = pl.Series([TS_PY_INT], dtype=pl.Int64())
# endregion array datetimes ------------------------------------------------------------

# region array timedeltas --------------------------------------------------------------
TD_NUMPY_DUR: NDArray[np.timedelta64] = np.array([TD_PY_DUR], dtype="timedelta64[ns]")
TD_NUMPY_FLOAT: NDArray[np.float64] = np.array([TD_PY_FLOAT], dtype="float64")
TD_NUMPY_INT: NDArray[np.int64] = np.array([TD_PY_INT], dtype="int64")
TD_PANDAS_NP_DUR: pd.Series = pd.Series([TD_PY_DUR], dtype="timedelta64[ns]")
TD_PANDAS_NP_FLOAT: pd.Series = pd.Series([TD_PY_FLOAT], dtype="float64")
TD_PANDAS_NP_INT: pd.Series = pd.Series([TD_PY_INT], dtype="int64")
TD_PANDAS_PA_DUR: pd.Series = pd.Series([TD_PY_DUR], dtype="duration[ns][pyarrow]")
TD_PANDAS_PA_FLOAT: pd.Series = pd.Series([TD_PY_FLOAT], dtype="float64[pyarrow]")
TD_PANDAS_PA_INT: pd.Series = pd.Series([TD_PY_INT], dtype="int64[pyarrow]")
TD_POLARS_DUR: pl.Series = pl.Series([TD_PY_DUR], dtype=pl.Duration())
TD_POLARS_FLOAT: pl.Series = pl.Series([TD_PY_FLOAT], dtype=pl.Float64())
TD_POLARS_INT: pl.Series = pl.Series([TD_PY_INT], dtype=pl.Int64())
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
TIMEDELTA_ARRAYS: dict[KEY, NumericalArray[DurationScalar]] = {
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
    "numpy[float]"     : TD_NP_FLOAT,
    "numpy[int]"       : TD_NP_INT,
    "numpy[time]"      : TD_NP_DUR,
    "pandas[np_time]"  : TD_PY_DUR,
    "pandas[np_float]" : TD_PY_FLOAT,
    "pandas[np_int]"   : TD_PY_INT,
    "pandas[pa_time]"  : TD_PY_DUR,
    "pandas[pa_float]" : TD_PY_FLOAT,
    "pandas[pa_int]"   : TD_PY_INT,
    "polars[time]"     : TD_PY_DUR,
    "polars[float]"    : TD_PY_FLOAT,
    "polars[int]"      : TD_PY_INT,
}  # fmt: skip
r"""Dictionary of compatible python timedelta values for each timedelta."""

TIMESTAMP_ARRAYS: dict[KEY, NumericalArray[TimestampScalar]] = {
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
    "numpy[float]"     : TS_NP_FLOAT,
    "numpy[int]"       : TS_NP_INT,
    "numpy[time]"      : TS_NP_DATE,
    "pandas[np_time]"  : TS_PY_DATE,
    "pandas[np_float]" : TS_PY_FLOAT,
    "pandas[np_int]"   : TS_PY_INT,
    "pandas[pa_time]"  : TS_PY_DATE,
    "pandas[pa_float]" : TS_PY_FLOAT,
    "pandas[pa_int]"   : TS_PY_INT,
    "polars[time]"     : TS_PY_DATE,
    "polars[float]"    : TS_PY_FLOAT,
    "polars[int]"      : TS_PY_INT,
}  # fmt: skip
r"""Dictionary of compatible python datetime values for each datetime."""
# endregion test data ------------------------------------------------------------------


@pytest.mark.parametrize("example", TEST_CASES)
def test_timedelta_arrays(example: KEY) -> None:
    td_array = TIMEDELTA_ARRAYS[example]
    td_scalar = TIMEDELTA_SCALARS[example]
    ts_scalar = TIMESTAMP_SCALARS[example]
    cls = type(td_array)

    # arithmetic
    assert type(td_array + td_scalar) is cls
    assert type(td_array - td_scalar) is cls
    assert type(ts_scalar + td_array) is cls
    assert type(ts_scalar - td_array) is cls
    # comparisons
    assert type(td_array < td_scalar) is cls


@pytest.mark.parametrize("example", TEST_CASES)
def test_timestamp_arrays(example: KEY) -> None:
    ts_array = TIMESTAMP_ARRAYS[example]
    td_scalar = TIMEDELTA_SCALARS[example]
    ts_scalar = TIMESTAMP_SCALARS[example]
    cls = type(ts_array)

    # comparisons
    assert type(ts_array < ts_scalar) is cls
    # arithmetic
    assert type(ts_array + td_scalar) is cls
    assert type(ts_array - td_scalar) is cls


if TYPE_CHECKING:
    TD_FLOAT_ARRAYS: dict[KEY, NumericalArray[float]] = {
        "numpy[float]"     : TD_NUMPY_FLOAT,
        "pandas[np_float]" : TD_PANDAS_NP_FLOAT,
        "pandas[pa_float]" : TD_PANDAS_PA_FLOAT,
        "polars[float]"    : TD_POLARS_FLOAT,
    }  # fmt: skip
    r"""Dictionary of float arrays."""

    TD_INT_ARRAYS: dict[KEY, NumericalArray[int]] = {
        "numpy[int]"     : TD_NUMPY_INT,
        "pandas[np_int]" : TD_PANDAS_NP_INT,
        "pandas[pa_int]" : TD_PANDAS_PA_INT,
        "polars[int]"    : TD_POLARS_INT,
    }  # fmt: skip
    r"""Dictionary of int arrays."""

    TD_TIME_ARRAYS: dict[KEY, NumericalArray[py_timedelta]] = {
        "numpy[time]"     : TD_NUMPY_DUR,
        "pandas[np_time]" : TD_PANDAS_NP_DUR,
        "pandas[pa_time]" : TD_PANDAS_PA_DUR,
        "polars[time]"    : TD_POLARS_DUR,
    }  # fmt: skip
    r"""Dictionary of timedelta arrays."""

    TS_FLOAT_ARRAYS: dict[KEY, NumericalArray[TimestampScalar[float]]] = {
        "numpy[float]"     : TS_NUMPY_FLOAT,
        "pandas[np_float]" : TS_PANDAS_NP_FLOAT,
        "pandas[pa_float]" : TS_PANDAS_PA_FLOAT,
        "polars[float]"    : TS_POLARS_FLOAT,
    }  # fmt: skip
    r"""Dictionary of float arrays."""

    TS_INT_ARRAYS: dict[KEY, NumericalArray[TimestampScalar[int]]] = {
        "numpy[int]"     : TS_NUMPY_INT,
        "pandas[np_int]" : TS_PANDAS_NP_INT,
        "pandas[pa_int]" : TS_PANDAS_PA_INT,
        "polars[int]"    : TS_POLARS_INT,
    }  # fmt: skip
    r"""Dictionary of int arrays."""

    TS_TIME_ARRAYS: dict[KEY, NumericalArray[TimestampScalar[py_timedelta]]] = {
        "numpy[time]"     : TS_NUMPY_DATE,
        "pandas[np_time]" : TS_PANDAS_NP_DATE,
        "pandas[pa_time]" : TS_PANDAS_PA_DATE,
        "polars[time]"    : TS_POLARS_DATE,
    }  # fmt: skip
    r"""Dictionary of datetime arrays."""

    TS_DATE_ARRAYS: dict[KEY, NumericalArray[py_datetime]] = {
        "numpy[time]"     : TS_NUMPY_DATE,
        "pandas[np_time]" : TS_PANDAS_NP_DATE,
        "pandas[pa_time]" : TS_PANDAS_PA_DATE,
        "polars[time]"    : TS_POLARS_DATE,
    }  # fmt: skip
    r"""Dictionary of datetime arrays."""
