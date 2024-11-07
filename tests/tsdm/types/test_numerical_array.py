r"""Test numerical arrays."""

from datetime import datetime as py_datetime, timedelta as py_timedelta

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch as pt

from tsdm.types.arrays import NumericalArray

BOOL_ARRAYS: dict[str, NumericalArray[bool]] = {
    "numpy[bool]"     : np.array([True], dtype=np.bool_),
    "pandas[np_bool]" : pd.Series([True], dtype=bool),
    "pandas[pa_bool]" : pd.Series([True], dtype="bool[pyarrow]"),
    "polars[bool]"    : pl.Series([True], dtype=pl.Boolean()),
    "torch[bool]"     : pt.tensor([True], dtype=pt.bool),
}  # fmt: skip
r"""Dictionary of bool arrays."""

INT_ARRAYS: dict[str, NumericalArray[int]] = {
    "numpy[int]"     : np.array([1], dtype=np.int64),
    "pandas[np_int]" : pd.Series([1], dtype=np.int64),
    "pandas[pa_int]" : pd.Series([1], dtype="int64[pyarrow]"),
    "polars[int]"    : pl.Series([1], dtype=pl.Int64()),
    "torch[int]"     : pt.tensor([1], dtype=pt.int64),
}  # fmt: skip
r"""Dictionary of int arrays."""

FLOAT_ARRAYS: dict[str, NumericalArray[float]] = {
    "numpy[float]"     : np.array([1.0], dtype=np.float64),
    "pandas[np_float]" : pd.Series([1.0], dtype=np.float64),
    "pandas[pa_float]" : pd.Series([1.0], dtype="float64[pyarrow]"),
    "polars[float]"    : pl.Series([1.0], dtype=pl.Float64()),
    "torch[float]"     : pt.tensor([1.0], dtype=pt.float64),
}  # fmt: skip
r"""Dictionary of float arrays."""

COMPLEX_ARRAYS: dict[str, NumericalArray[complex]] = {
    "numpy[complex]"     : np.array([1 + 1j], dtype=np.complex128),
    "torch[complex]"     : pt.tensor([1 + 1j], dtype=pt.complex128),
    "pandas[np_complex]" : pd.Series([1 + 1j], dtype=np.complex128),
}  # fmt: skip
r"""Dictionary of complex arrays."""

TIME_ARRAYS: dict[str, NumericalArray[py_timedelta]] = {
    "numpy[time]"     : np.array([py_timedelta(days=1)], dtype="timedelta64[ns]"),
    "pandas[np_time]" : pd.Series([py_timedelta(days=1)], dtype="timedelta64[ns]"),
    "pandas[pa_time]" : pd.Series([py_timedelta(days=1)], dtype=pd.ArrowDtype(pa.duration("s"))),
    "polars[time]"    : pl.Series([py_timedelta(days=1)], dtype=pl.Time()),
}  # fmt: skip
r"""Dictionary of timedelta arrays."""

DATE_ARRAYS: dict[str, NumericalArray[py_datetime]] = {
    "numpy[date]"     : np.array([py_datetime(2021, 1, 1)], dtype="datetime64[ns]"),
    "pandas[np_date]" : pd.Series([py_datetime(2021, 1, 1)], dtype="datetime64[ns]"),
    "pandas[pa_date]" : pd.Series([py_datetime(2021, 1, 1)], dtype=pd.ArrowDtype(pa.timestamp("ns"))),
    "polars[date]"    : pl.Series([py_datetime(2021, 1, 1)], dtype=pl.Date()),
}  # fmt: skip
r"""Dictionary of datetime arrays."""


@pytest.mark.parametrize("example", BOOL_ARRAYS)
def test_bool_array(example: str):
    r"""Test bool arrays."""
    array = BOOL_ARRAYS[example]
    assert isinstance(array, NumericalArray)
    cls = type(array)

    as_bool = True

    # test __and__
    assert type(array & array) is cls
    assert type(array & as_bool) is cls
    # test __or__
    assert type(array | array) is cls
    assert type(array | as_bool) is cls
    # test __xor__
    assert type(array ^ array) is cls
    assert type(array ^ as_bool) is cls


@pytest.mark.parametrize("example", INT_ARRAYS)
def test_int_array(example: str):
    r"""Test int arrays."""
    array = INT_ARRAYS[example]
    assert isinstance(array, NumericalArray)
    cls = type(array)

    as_int = 1
    as_bool = True

    # test __abs__
    assert type(abs(array)) is cls
    # test __neg__
    assert type(-array) is cls
    # test __pos__
    assert type(+array) is cls

    # test __add__
    assert type(array + array) is cls
    assert type(array + as_bool) is cls
    assert type(array + as_int) is cls
    # assert isinstance(array + as_float, FloatScalar)
    # assert isinstance(array + as_complex, ComplexScalar)
    # test __sub__
    assert type(array - array) is cls
    # assert type(array - as_bool) is cls  # not supported by torch
    assert type(array - as_int) is cls
    # assert isinstance(array - as_float, FloatScalar)
    # assert isinstance(array - as_complex, ComplexScalar)
    # test __mul__
    assert type(array * array) is cls
    assert type(array * as_bool) is cls
    assert type(array * as_int) is cls
    # assert isinstance(array * as_float, FloatScalar)
    # assert isinstance(array * as_complex, ComplexScalar)
    # test __pow__
    assert type(array**array) is cls
    assert type(array**as_bool) is cls
    assert type(array**as_int) is cls
    # assert isinstance(array**as_float, FloatScalar)
    # assert isinstance(array**as_complex, ComplexScalar)
    # test __mod__
    assert type(array % array) is cls
    assert type(array % as_bool) is cls
    assert type(array % as_int) is cls
    # assert isinstance(array % as_float, FloatScalar)
    # assert isinstance(array % as_complex, ComplexScalar)  # nonsensical
    # test __floordiv__
    assert type(array // array) is cls
    assert type(array // as_bool) is cls
    assert type(array // as_int) is cls
    # assert isinstance(array // as_float, FloatScalar)
    # assert isinstance(array // as_complex, ComplexScalar)  # nonsensical


@pytest.mark.parametrize("example", FLOAT_ARRAYS)
def test_float_array(example: str):
    r"""Test float arrays."""
    array = FLOAT_ARRAYS[example]
    assert isinstance(array, NumericalArray)
    cls = type(array)

    as_float = 1.0

    # test __abs__
    assert type(abs(array)) is cls
    # test __neg__
    assert type(-array) is cls
    # test __pos__
    assert type(+array) is cls

    # test __add__
    assert type(array + array) is cls
    assert type(array + as_float) is cls
    # assert isinstance(array + as_complex, ComplexScalar)
    # test __sub__
    assert type(array - array) is cls
    assert type(array - as_float) is cls
    # assert isinstance(array - as_complex, ComplexScalar)
    # test __mul__
    assert type(array * array) is cls
    assert type(array * as_float) is cls
    # assert isinstance(array * as_complex, ComplexScalar)
    # test __truediv__
    assert type(array / array) is cls
    assert type(array / as_float) is cls
    # assert isinstance(array / as_complex, ComplexScalar)
    # test __pow__
    assert type(array**array) is cls
    assert type(array**as_float) is cls
    # assert isinstance(array**as_complex, ComplexScalar)
    # test __floordiv__
    assert type(array // array) is cls
    assert type(array // as_float) is cls
    # assert isinstance(array // as_complex, ComplexScalar)  # nonsensical


@pytest.mark.parametrize("example", COMPLEX_ARRAYS)
def test_complex_array(example: str):
    r"""Test complex arrays."""
    array = COMPLEX_ARRAYS[example]
    assert isinstance(array, NumericalArray)
    cls = type(array)

    as_complex = complex(1.0)

    # test __abs__
    assert type(abs(array)) is cls
    # test __neg__
    assert type(-array) is cls
    # test __pos__
    assert type(+array) is cls

    # test __add__
    assert type(array + array) is cls
    assert type(array + as_complex) is cls
    # test __sub__
    assert type(array - array) is cls
    assert type(array - as_complex) is cls
    # test __mul__
    assert type(array * array) is cls
    assert type(array * as_complex) is cls
    # test __truediv__
    assert type(array / array) is cls
    assert type(array / as_complex) is cls
    # test __pow__
    assert type(array**array) is cls
    assert type(array**as_complex) is cls


@pytest.mark.parametrize("example", TIME_ARRAYS)
def test_time_array(example: str):
    r"""Test timedelta arrays."""
    array = TIME_ARRAYS[example]
    assert isinstance(array, NumericalArray)
    cls = type(array)

    # test comparisons
    assert type(array > array) is cls
    assert type(array < array) is cls
    assert type(array >= array) is cls
    assert type(array <= array) is cls
    assert type(array == array) is cls
    assert type(array != array) is cls

    # test __abs__
    assert type(abs(array)) is cls
    # test __neg__
    assert type(-array) is cls
    # test __pos__
    assert type(+array) is cls

    # test __add__
    assert type(array + array) is cls
    # test __sub__
    assert type(array - array) is cls
    # test __mul__
    assert type(array * 2) is cls
    # test modulo
    # assert type(array % array) is cls
    # test __floordiv__
    # assert type(array // 2) is cls
    assert type(array // array) is cls
    # test __truediv__
    assert type(array / 2) is cls


@pytest.mark.parametrize("example", DATE_ARRAYS)
def test_date_array(example: str):
    r"""Test datetime arrays."""
    array = DATE_ARRAYS[example]
    assert isinstance(array, NumericalArray)
    cls = type(array)

    # test comparisons
    assert type(array > array) is cls
    assert type(array < array) is cls
    assert type(array >= array) is cls
    assert type(array <= array) is cls
    assert type(array == array) is cls
    assert type(array != array) is cls

    # test __sub__
    zero = array - array
    assert isinstance(zero, NumericalArray)
    # test __add__
    assert type(array + zero) is cls
