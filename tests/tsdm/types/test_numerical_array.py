r"""Test numerical arrays."""

from datetime import datetime as py_datetime, timedelta as py_timedelta

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch as pt

from tsdm.types.linalg import (
    BooleanArray,
    ComplexArray,
    DatetimeArray,
    FloatArray,
    IntegerArray,
    NumericalArray,
    TimedeltaArray,
)

BOOL_ARRAYS: dict[str, BooleanArray] = {
    "numpy[bool]"     : np.array([True], dtype=np.bool_),
    "pandas[np_bool]" : pd.Series([True], dtype=bool),
    "pandas[pa_bool]" : pd.Series([True], dtype="bool[pyarrow]"),
    "polars[bool]"    : pl.Series([True], dtype=pl.Boolean()),
    "torch[bool]"     : pt.tensor([True], dtype=pt.bool),
}  # fmt: skip
r"""Dictionary of bool arrays."""

INT_ARRAYS: dict[str, IntegerArray] = {
    "numpy[int]"     : np.array([1], dtype=np.int64),  # pyright: ignore[reportAssignmentType]
    "pandas[np_int]" : pd.Series([1], dtype=np.int64),
    "pandas[pa_int]" : pd.Series([1], dtype="int64[pyarrow]"),
    "polars[int]"    : pl.Series([1], dtype=pl.Int64()),
    "torch[int]"     : pt.tensor([1], dtype=pt.int64),
}  # fmt: skip
r"""Dictionary of int arrays."""

FLOAT_ARRAYS: dict[str, FloatArray] = {
    "numpy[float]"     : np.array([1.0], dtype=np.float64),  # pyright: ignore[reportAssignmentType]
    "pandas[np_float]" : pd.Series([1.0], dtype=np.float64),
    "pandas[pa_float]" : pd.Series([1.0], dtype="float64[pyarrow]"),
    "polars[float]"    : pl.Series([1.0], dtype=pl.Float64()),
    "torch[float]"     : pt.tensor([1.0], dtype=pt.float64),
}  # fmt: skip
r"""Dictionary of float arrays."""

COMPLEX_ARRAYS: dict[str, ComplexArray] = {
    "numpy[complex]"     : np.array([1 + 1j], dtype=np.complex128),  # pyright: ignore[reportAssignmentType]
    "torch[complex]"     : pt.tensor([1 + 1j], dtype=pt.complex128),
    "pandas[np_complex]" : pd.Series([1 + 1j], dtype=np.complex128),
}  # fmt: skip
r"""Dictionary of complex arrays."""

TIME_ARRAYS: dict[str, TimedeltaArray] = {
    "numpy[time]"     : np.array([py_timedelta(days=1)], dtype="timedelta64[ns]"),
    "pandas[np_time]" : pd.Series([py_timedelta(days=1)], dtype="timedelta64[ns]"),
    "pandas[pa_time]" : pd.Series([py_timedelta(days=1)], dtype=pd.ArrowDtype(pa.duration("s"))),
    "polars[time]"    : pl.Series([py_timedelta(days=1)], dtype=pl.Time()),
}  # fmt: skip
r"""Dictionary of timedelta arrays."""

DATE_ARRAYS: dict[str, DatetimeArray] = {
    "numpy[date]"     : np.array([py_datetime(2021, 1, 1)], dtype="datetime64[ns]"),
    "pandas[np_date]" : pd.Series([py_datetime(2021, 1, 1)], dtype="datetime64[ns]"),
    "pandas[pa_date]" : pd.Series([py_datetime(2021, 1, 1)], dtype=pd.ArrowDtype(pa.timestamp("ns"))),
    "polars[date]"    : pl.Series([py_datetime(2021, 1, 1)], dtype=pl.Date()),
}  # fmt: skip
r"""Dictionary of datetime arrays."""


BOOL: bool = bool(1)
INT: int = int(1.0)
FLOAT: float = float(1)
COMPLEX: complex = complex(0 + 1j)
DATETIME: py_datetime = py_datetime(2021, 1, 1)
TIMEDELTA: py_timedelta = py_timedelta(days=1)


@pytest.mark.parametrize("example", BOOL_ARRAYS)
def test_bool_array(example: str) -> None:
    r"""Test bool arrays."""
    array = BOOL_ARRAYS[example]
    cls = type(array)
    assert isinstance(array, BooleanArray)

    # fmt: off
    # test unary operations
    assert type(~array) is cls  # __invert__
    # test vector operations
    assert type(array == array) is cls  # __eq__
    assert type(array != array) is cls  # __ne__
    assert type(array < array) is cls   # __lt__
    assert type(array <= array) is cls  # __le__
    assert type(array > array) is cls   # __gt__
    assert type(array >= array) is cls  # __ge__
    assert type(array & array) is cls   # __and__
    assert type(array | array) is cls   # __or__
    assert type(array ^ array) is cls   # __xor__
    # test scalar operations (bool)
    assert type(array == BOOL) is cls   # __eq__
    assert type(array != BOOL) is cls   # __ne__
    # assert type(array < BOOL) is cls    # __lt__
    # assert type(array <= BOOL) is cls   # __le__
    # assert type(array > BOOL) is cls    # __gt__
    # assert type(array >= BOOL) is cls   # __ge__
    assert type(array & BOOL) is cls    # __and__
    assert type(array | BOOL) is cls    # __or__
    assert type(array ^ BOOL) is cls    # __xor__
    # test reverse scalar operations (bool)
    # assert type(BOOL == array) is cls   # __eq__
    # assert type(BOOL != array) is cls   # __ne__
    # assert type(BOOL < array) is cls    # __lt__
    # assert type(BOOL <= array) is cls   # __le__
    # assert type(BOOL > array) is cls    # __gt__
    # assert type(BOOL >= array) is cls   # __ge__
    assert type(BOOL & array) is cls    # __and__
    assert type(BOOL | array) is cls    # __or__
    assert type(BOOL ^ array) is cls    # __xor__
    # fmt: on


@pytest.mark.parametrize("example", INT_ARRAYS)
def test_int_array(example: str) -> None:
    r"""Test int arrays."""
    array = INT_ARRAYS[example]
    cls = type(array)
    assert isinstance(array, IntegerArray)

    # fmt: off
    # test unary operations
    assert type(abs(array)) is cls      # __abs__
    assert type(-array) is cls          # __neg__
    assert type(+array) is cls          # __pos__
    assert type(~array) is cls          # __invert__
    # test vector operations
    assert type(array == array) is cls  # __eq__
    assert type(array != array) is cls  # __ne__
    assert type(array < array) is cls   # __lt__
    assert type(array <= array) is cls  # __le__
    assert type(array > array) is cls   # __gt__
    assert type(array >= array) is cls  # __ge__
    assert type(array + array) is cls   # __add__
    assert type(array - array) is cls   # __sub__
    assert type(array * array) is cls   # __mul__
    assert type(array**array) is cls    # __pow__
    assert type(array // array) is cls  # __floordiv__
    assert type(array % array) is cls   # __mod__
    # test scalar operations (int)
    assert type(array == INT) is cls    # __eq__
    assert type(array != INT) is cls    # __ne__
    assert type(array < INT) is cls     # __lt__
    assert type(array <= INT) is cls    # __le__
    assert type(array > INT) is cls     # __gt__
    assert type(array >= INT) is cls    # __ge__
    assert type(array + INT) is cls     # __add__
    assert type(array - INT) is cls     # __sub__
    assert type(array * INT) is cls     # __mul__
    assert type(array**INT) is cls      # __pow__
    assert type(array // INT) is cls    # __floordiv__
    assert type(array % INT) is cls     # __mod__
    # assert type(array & INT) is cls     # __and__
    # assert type(array | INT) is cls     # __or__
    # assert type(array ^ INT) is cls     # __xor__
    # test reverse scalar operations (int)
    # assert type(INT == array) is cls    # __eq__
    # assert type(INT != array) is cls    # __ne__
    assert type(INT < array) is cls     # __lt__
    assert type(INT <= array) is cls    # __le__
    assert type(INT > array) is cls     # __gt__
    assert type(INT >= array) is cls    # __ge__
    assert type(INT + array) is cls     # __add__
    assert type(INT - array) is cls     # __sub__
    assert type(INT * array) is cls     # __mul__
    assert type(INT**array) is cls      # __pow__
    assert type(INT // array) is cls    # __floordiv__
    assert type(INT % array) is cls     # __mod__
    # assert type(INT & array) is cls     # __and__
    # assert type(INT | array) is cls     # __or__
    # assert type(INT ^ array) is cls     # __xor__
    # test scalar operations (float)
    assert type(array == FLOAT) is cls    # __eq__
    assert type(array != FLOAT) is cls    # __ne__
    assert type(array < FLOAT) is cls     # __lt__
    assert type(array <= FLOAT) is cls    # __le__
    assert type(array > FLOAT) is cls     # __gt__
    assert type(array >= FLOAT) is cls    # __ge__
    # test scalar operations (float)
    # assert type(FLOAT == array) is cls    # __eq__
    # assert type(FLOAT != array) is cls    # __ne__
    assert type(FLOAT < array) is cls     # __lt__
    assert type(FLOAT <= array) is cls    # __le__
    assert type(FLOAT > array) is cls     # __gt__
    assert type(FLOAT >= array) is cls    # __ge__
    # fmt: on


@pytest.mark.parametrize("example", FLOAT_ARRAYS)
def test_float_array(example: str) -> None:
    r"""Test float arrays."""
    array = FLOAT_ARRAYS[example]
    cls = type(array)
    assert isinstance(array, FloatArray)

    # fmt: off
    # test unary operations
    assert type(abs(array)) is cls      # __abs__
    assert type(-array) is cls          # __neg__
    assert type(+array) is cls          # __pos__
    # test vector operations
    assert type(array == array) is cls  # __eq__
    assert type(array != array) is cls  # __ne__
    assert type(array < array) is cls   # __lt__
    assert type(array <= array) is cls  # __le__
    assert type(array > array) is cls   # __gt__
    assert type(array >= array) is cls  # __ge__
    assert type(array + array) is cls   # __add__
    assert type(array - array) is cls   # __sub__
    assert type(array * array) is cls   # __mul__
    assert type(array**array) is cls    # __pow__
    assert type(array / array) is cls   # __truediv__
    assert type(array // array) is cls  # __floordiv__
    assert type(array % array) is cls   # __mod__
    # test scalar operations (float)
    assert type(array == FLOAT) is cls  # __eq__
    assert type(array != FLOAT) is cls  # __ne__
    assert type(array < FLOAT) is cls   # __lt__
    assert type(array <= FLOAT) is cls  # __le__
    assert type(array > FLOAT) is cls   # __gt__
    assert type(array >= FLOAT) is cls  # __ge__
    assert type(array + FLOAT) is cls   # __add__
    assert type(array - FLOAT) is cls   # __sub__
    assert type(array * FLOAT) is cls   # __mul__
    assert type(array**FLOAT) is cls    # __pow__
    assert type(array / FLOAT) is cls   # __truediv__
    assert type(array // FLOAT) is cls  # __floordiv__
    assert type(array % FLOAT) is cls   # __mod__
    # test reverse scalar operations (float)
    # assert type(FLOAT == array) is cls  # __eq__
    # assert type(FLOAT != array) is cls  # __ne__
    assert type(FLOAT < array) is cls   # __lt__
    assert type(FLOAT <= array) is cls  # __le__
    assert type(FLOAT > array) is cls   # __gt__
    assert type(FLOAT >= array) is cls  # __ge__
    assert type(FLOAT + array) is cls   # __add__
    assert type(FLOAT - array) is cls   # __sub__
    assert type(FLOAT * array) is cls   # __mul__
    assert type(FLOAT**array) is cls    # __pow__
    assert type(FLOAT / array) is cls   # __truediv__
    assert type(FLOAT // array) is cls  # __floordiv__
    assert type(FLOAT % array) is cls   # __mod__
    # test scalar operations (int)
    assert type(array == INT) is cls    # __eq__
    assert type(array != INT) is cls    # __ne__
    assert type(array < INT) is cls     # __lt__
    assert type(array <= INT) is cls    # __le__
    assert type(array > INT) is cls     # __gt__
    assert type(array >= INT) is cls    # __ge__
    assert type(array + INT) is cls     # __add__
    assert type(array - INT) is cls     # __sub__
    assert type(array * INT) is cls     # __mul__
    assert type(array**INT) is cls      # __pow__
    assert type(array / INT) is cls     # __truediv__
    assert type(array // INT) is cls    # __floordiv__
    assert type(array % INT) is cls     # __mod__
    # test reverse scalar operations (int)
    # assert type(INT == array) is cls    # __eq__
    # assert type(INT != array) is cls    # __ne__
    assert type(INT < array) is cls     # __lt__
    assert type(INT <= array) is cls    # __le__
    assert type(INT > array) is cls     # __gt__
    assert type(INT >= array) is cls    # __ge__
    assert type(INT + array) is cls     # __add__
    assert type(INT - array) is cls     # __sub__
    assert type(INT * array) is cls     # __mul__
    assert type(INT**array) is cls      # __pow__
    assert type(INT / array) is cls     # __truediv__
    assert type(INT // array) is cls    # __floordiv__
    assert type(INT % array) is cls     # __mod__
    # fmt: on


@pytest.mark.parametrize("example", COMPLEX_ARRAYS)
def test_complex_array(example: str) -> None:
    r"""Test complex arrays."""
    array = COMPLEX_ARRAYS[example]
    cls = type(array)
    assert isinstance(array, ComplexArray)

    # fmt: off
    # test unary operations
    assert type(abs(array)) is cls      # __abs__
    assert type(-array) is cls          # __neg__
    assert type(+array) is cls          # __pos__
    # test vector operations
    assert type(array == array) is cls  # __eq__
    assert type(array != array) is cls  # __ne__
    assert type(array + array) is cls   # __add__
    assert type(array - array) is cls   # __sub__
    assert type(array * array) is cls   # __mul__
    assert type(array / array) is cls   # __truediv__
    assert type(array**array) is cls    # __pow__
    # test scalar operations (complex)
    assert type(array == COMPLEX) is cls  # __eq__
    assert type(array != COMPLEX) is cls  # __ne__
    assert type(array + COMPLEX) is cls   # __add__
    assert type(array - COMPLEX) is cls   # __sub__
    assert type(array * COMPLEX) is cls   # __mul__
    assert type(array / COMPLEX) is cls   # __truediv__
    assert type(array**COMPLEX) is cls    # __pow__
    # test reverse scalar operations (complex)
    # assert type(COMPLEX == array) is cls  # __eq__
    # assert type(COMPLEX != array) is cls  # __ne__
    assert type(COMPLEX + array) is cls   # __add__
    assert type(COMPLEX - array) is cls   # __sub__
    assert type(COMPLEX * array) is cls   # __mul__
    assert type(COMPLEX / array) is cls   # __truediv__
    assert type(COMPLEX**array) is cls    # __pow__
    # test scalar operations (float)
    assert type(array == FLOAT) is cls  # __eq__
    assert type(array != FLOAT) is cls  # __ne__
    assert type(array + FLOAT) is cls   # __add__
    assert type(array - FLOAT) is cls   # __sub__
    assert type(array * FLOAT) is cls   # __mul__
    assert type(array / FLOAT) is cls   # __truediv__
    assert type(array**FLOAT) is cls    # __pow__
    # test reverse scalar operations (float)
    # assert type(FLOAT == array) is cls  # __eq__
    # assert type(FLOAT != array) is cls  # __ne__
    assert type(FLOAT + array) is cls   # __add__
    assert type(FLOAT - array) is cls   # __sub__
    assert type(FLOAT * array) is cls   # __mul__
    assert type(FLOAT / array) is cls   # __truediv__
    assert type(FLOAT**array) is cls    # __pow__
    # test scalar operations (int)
    assert type(array == INT) is cls  # __eq__
    assert type(array != INT) is cls  # __ne__
    assert type(array + INT) is cls   # __add__
    assert type(array - INT) is cls   # __sub__
    assert type(array * INT) is cls   # __mul__
    assert type(array / INT) is cls   # __truediv__
    assert type(array**INT) is cls    # __pow__
    # test reverse scalar operations (int)
    # assert type(INT == array) is cls  # __eq__
    # assert type(INT != array) is cls  # __ne__
    assert type(INT + array) is cls   # __add__
    assert type(INT - array) is cls   # __sub__
    assert type(INT * array) is cls   # __mul__
    assert type(INT / array) is cls   # __truediv__
    assert type(INT**array) is cls    # __pow__
    # fmt: on


@pytest.mark.parametrize("example", TIME_ARRAYS)
def test_timedelta_array(example: str) -> None:
    r"""Test timedelta arrays."""
    array = TIME_ARRAYS[example]
    cls = type(array)
    assert isinstance(array, NumericalArray)

    # fmt: off
    # test unary operations
    assert type(abs(array)) is cls           # __abs__
    assert type(-array) is cls               # __neg__
    assert type(+array) is cls               # __pos__
    # test vector operations
    assert type(array == array) is cls       # __eq__
    assert type(array != array) is cls       # __ne__
    assert type(array < array) is cls        # __lt__
    assert type(array <= array) is cls       # __le__
    assert type(array > array) is cls        # __gt__
    assert type(array >= array) is cls       # __ge__

    assert type(array + array) is cls        # __add__
    assert type(array - array) is cls        # __sub__

    # test scalar operations (timedelta)
    assert type(array == TIMEDELTA) is cls   # __eq__
    assert type(array != TIMEDELTA) is cls   # __ne__
    assert type(array < TIMEDELTA) is cls    # __lt__
    assert type(array <= TIMEDELTA) is cls   # __le__
    assert type(array > TIMEDELTA) is cls    # __gt__
    assert type(array >= TIMEDELTA) is cls   # __ge__

    # test reverse scalar operations (timedelta)
    # assert type(TIMEDELTA == array) is cls   # __eq__
    # assert type(TIMEDELTA != array) is cls   # __ne__
    assert type(TIMEDELTA < array) is cls    # __lt__
    assert type(TIMEDELTA <= array) is cls   # __le__
    assert type(TIMEDELTA > array) is cls    # __gt__
    assert type(TIMEDELTA >= array) is cls   # __ge__

    # test scalar operations (int)
    assert type(array * 2) is cls            # __mul__
    assert type(2 * array) is cls            # __rmul__
    assert type(array / 2) is cls            # __truediv__
    # fmt: on


@pytest.mark.parametrize("example", DATE_ARRAYS)
def test_datetime_array(example: str) -> None:
    r"""Test datetime arrays."""
    array = DATE_ARRAYS[example]
    cls = type(array)
    ZERO = array - array
    assert isinstance(array, DatetimeArray)

    # fmt: off
    # test vector operations
    assert type(array == array) is cls      # __eq__
    assert type(array != array) is cls      # __ne__
    assert type(array < array) is cls       # __lt__
    assert type(array <= array) is cls      # __le__
    assert type(array > array) is cls       # __gt__
    assert type(array >= array) is cls      # __ge__

    assert type(array + ZERO) is cls        # __add__
    assert type(array - array) is cls       # __sub__

    # test scalar operations (datetime)
    assert type(array == DATETIME) is cls   # __eq__
    assert type(array != DATETIME) is cls   # __ne__
    assert type(array < DATETIME) is cls    # __lt__
    assert type(array <= DATETIME) is cls   # __le__
    assert type(array > DATETIME) is cls    # __gt__
    assert type(array >= DATETIME) is cls   # __ge__

    assert type(array + TIMEDELTA) is cls   # __add__
    assert type(array - DATETIME) is cls    # __sub__

    # test reverse scalar operations (datetime)
    # assert type(DATETIME == array) is cls   # __eq__
    # assert type(DATETIME != array) is cls   # __ne__
    assert type(DATETIME < array) is cls    # __lt__
    assert type(DATETIME <= array) is cls   # __le__
    assert type(DATETIME > array) is cls    # __gt__
    assert type(DATETIME >= array) is cls   # __ge__

    assert type(TIMEDELTA + array) is cls   # __add__
    assert type(DATETIME - array) is cls    # __sub__
    # fmt: on
