r"""Test numerical arrays."""
# mypy: disable-error-code="unreachable"

from datetime import datetime as py_datetime, timedelta as py_timedelta

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch as pt

from tests import pytest_xfail
from tsdm.testing import assert_protocol
from tsdm.types.numerical import (
    BooleanArray,
    ComplexArray,
    FloatArray,
    IntegerArray,
    SpanLikeArray,
    TimeLikeArray,
)

_pa_bool = pd.ArrowDtype(pa.bool_())
_pa_float64 = pd.ArrowDtype(pa.float64())
_pa_int64 = pd.ArrowDtype(pa.int64())
_pa_duration_ns = pd.ArrowDtype(pa.duration("ns"))
_pa_timestamp_ns = pd.ArrowDtype(pa.timestamp("ns"))

# FIXME: https://github.com/pola-rs/polars/issues/23132

_BOOLS: list[bool] = [True, False, True, False]
_INTS: list[int] = [1, 2, 3]
_FLOATS: list[float] = [1.0, 0.0, -1.0]
_COMPLEX: list[complex] = [1j, 0.0, -1j]
_DATETIMES: list[py_datetime] = [
    py_datetime(2021, 1, 1),
    py_datetime(2021, 1, 2),
    py_datetime(2021, 1, 3),
]
_TIMEDELTAS: list[py_timedelta] = [
    py_timedelta(days=1),
    py_timedelta(days=2),
    py_timedelta(days=3),
]


BOOL_ARRAYS: dict[str, BooleanArray] = {
    "numpy[bool]"     : np.array(_BOOLS, dtype=np.bool_),
    "pandas[np_bool]" : pd.Series(_BOOLS, dtype=bool),
    "pandas[pa_bool]" : pd.Series(_BOOLS, dtype=_pa_bool),
    "polars[bool]"    : pl.Series(_BOOLS, dtype=pl.Boolean()),
    "torch[bool]"     : pt.tensor(_BOOLS, dtype=pt.bool),
}  # fmt: skip
r"""Dictionary of bool arrays."""

INT_ARRAYS: dict[str, IntegerArray] = {
    "numpy[int]"     : np.array(_INTS, dtype=np.int64),
    "pandas[np_int]" : pd.Series(_INTS, dtype=np.int64),
    "pandas[pa_int]" : pd.Series(_INTS, dtype=_pa_int64),
    "polars[int]"    : pl.Series(_INTS, dtype=pl.Int64()),
    "torch[int]"     : pt.tensor(_INTS, dtype=pt.int64),
}  # fmt: skip
r"""Dictionary of int arrays."""

FLOAT_ARRAYS: dict[str, FloatArray] = {
    "numpy[float]"     : np.array(_FLOATS, dtype=np.float64),
    "pd_series[np_float]" : pd.Series(_FLOATS, dtype=np.float64),
    "pd_series[pa_float]" : pd.Series(_FLOATS, dtype=_pa_float64),
    "polars[float]"    : pl.Series(_FLOATS, dtype=pl.Float64()),
    "torch[float]"     : pt.tensor(_FLOATS, dtype=pt.float64),
}  # fmt: skip
r"""Dictionary of float arrays."""

COMPLEX_ARRAYS: dict[str, ComplexArray] = {
    "numpy[complex]"     : np.array(_COMPLEX, dtype=np.complex128),
    "torch[complex]"     : pt.tensor(_COMPLEX, dtype=pt.complex128),
    "pandas[np_complex]" : pd.Series(_COMPLEX, dtype=np.complex128),
}  # fmt: skip
r"""Dictionary of complex arrays."""

TIME_ARRAYS: dict[str, SpanLikeArray] = {
    "numpy[time]"     : np.array(_TIMEDELTAS, dtype="timedelta64[ns]"),
    "pandas[np_time]" : pd.Series(_TIMEDELTAS, dtype="timedelta64[ns]"),
    "pandas[pa_time]" : pd.Series(_TIMEDELTAS, dtype=_pa_duration_ns),
    "polars[time]"    : pl.Series(_TIMEDELTAS, dtype=pl.Time()),
}  # fmt: skip
r"""Dictionary of timedelta arrays."""

DATE_ARRAYS: dict[str, TimeLikeArray] = {
    "numpy[date]"     : np.array(_DATETIMES, dtype="datetime64[ns]"),
    "pandas[np_date]" : pd.Series(_DATETIMES, dtype="datetime64[ns]"),
    "pandas[pa_date]" : pd.Series(_DATETIMES, dtype=_pa_timestamp_ns),
    "polars[date]"    : pl.Series(_DATETIMES, dtype=pl.Date()),
}  # fmt: skip
r"""Dictionary of datetime arrays."""


BOOL: bool = bool(1)
INT: int = int(1.0)
FLOAT: float = float(1)
COMPLEX: complex = complex(0 + 1j)
DATETIME: py_datetime = py_datetime(2021, 1, 1)
TIMEDELTA: py_timedelta = py_timedelta(days=1)


def test_joint_interface_floatarray() -> None:
    shared_attrs = set.intersection(*(set(dir(s)) for s in FLOAT_ARRAYS.values()))
    other_attrs = shared_attrs - set(dir(FloatArray))
    print(other_attrs)


@pytest.mark.parametrize("example", BOOL_ARRAYS)
def test_bool_array(example: str) -> None:
    r"""Test bool arrays."""
    array = BOOL_ARRAYS[example]
    cls = type(array)

    # fmt: off
    assert type( ~array ) is cls  # __invert__

    assert type( array == array ) is cls  # __eq__(self)
    assert type( array == BOOL  ) is cls  # __eq__(bool)
    assert type( BOOL  == array ) is cls  # __eq__(bool)  # type: ignore[unreachable]
    assert type( array != array ) is cls  # __ne__(self)
    assert type( array != BOOL  ) is cls  # __ne__(bool)
    assert type( BOOL  != array ) is cls  # __ne__(bool)

    with pytest_xfail("polars", strict=(example == "polars[bool]")):
        assert type( array <  array ) is cls  # __lt__(self)
        assert type( array <  BOOL  ) is cls  # __lt__(bool)
        assert type( BOOL  <  array ) is cls  # __lt__(bool)

        assert type( array <= array ) is cls  # __le__(self)
        assert type( array <= BOOL  ) is cls  # __le__(bool)
        assert type( BOOL  <= array ) is cls  # __le__(bool)

        assert type( array >  array ) is cls  # __gt__(self)
        assert type( array >  BOOL  ) is cls  # __gt__(bool)
        assert type( BOOL  >  array ) is cls  # __gt__(bool)

        assert type( array >= array ) is cls  # __ge__(self)
        assert type( array >= BOOL  ) is cls  # __ge__(bool)
        assert type( BOOL  >= array ) is cls  # __ge__(bool)

    assert type( array &  array ) is cls  # __and__(self)
    assert type( array &  BOOL  ) is cls  # __and__(bool)
    assert type( BOOL  &  array ) is cls  # __rand__(bool)

    assert type( array |  array ) is cls  # __or__(self)
    assert type( array |  BOOL  ) is cls  # __or__(bool)
    assert type( BOOL  |  array ) is cls  # __ror__(bool)

    assert type( array ^  array ) is cls  # __xor__(self)
    assert type( array ^  BOOL  ) is cls  # __xor__(bool)
    assert type( BOOL  ^  array ) is cls  # __rxor__(bool)
    # fmt: on


@pytest.mark.parametrize("case", INT_ARRAYS)
def test_int_array(case: str) -> None:
    r"""Test int arrays."""
    array = INT_ARRAYS[case]
    cls = type(array)

    # test interface
    assert array.min() == 1
    assert array.max() == 3
    assert type(array.clip(-1, 1)) is cls

    # fmt: off
    assert type( abs(array) ) is cls      # __abs__
    assert type(    -array  ) is cls      # __neg__
    assert type(    +array  ) is cls      # __pos__
    assert type(    ~array  ) is cls      # __invert__

    assert type( array == array ) is cls  # __eq__(self)
    assert type( array == INT   ) is cls  # __eq__(int)
    assert type( array == FLOAT ) is cls  # __eq__(float)
    assert type( INT   == array ) is cls  # __eq__(int)
    assert type( FLOAT == array ) is cls  # __eq__(float)

    assert type( array != array ) is cls  # __ne__(self)
    assert type( array != INT   ) is cls  # __ne__(int)
    assert type( array != FLOAT ) is cls  # __ne__(float)
    assert type( INT   != array ) is cls  # __ne__(int)
    assert type( FLOAT != array ) is cls  # __ne__(float)

    assert type( array <  array ) is cls  # __lt__(self)
    assert type( array <  INT   ) is cls  # __lt__(int)
    assert type( array <  FLOAT ) is cls  # __lt__(float)
    assert type( INT   <  array ) is cls  # __lt__(int)
    assert type( FLOAT <  array ) is cls  # __lt__(float)

    assert type( array <= array ) is cls  # __le__(self)
    assert type( array <= INT   ) is cls  # __le__(int)
    assert type( array <= FLOAT ) is cls  # __le__(float)
    assert type( INT   <= array ) is cls  # __le__(int)
    assert type( FLOAT <= array ) is cls  # __le__(float)

    assert type( array >  array ) is cls  # __gt__(self)
    assert type( array >  INT   ) is cls  # __gt__(int)
    assert type( array >  FLOAT ) is cls  # __gt__(float)
    assert type( INT   >  array ) is cls  # __gt__(int)
    assert type( FLOAT >  array ) is cls  # __gt__(float)

    assert type( array >= array ) is cls  # __ge__(self)
    assert type( array >= INT   ) is cls  # __ge__(int)
    assert type( array >= FLOAT ) is cls  # __ge__(float)
    assert type( INT   >= array ) is cls  # __ge__(int)
    assert type( FLOAT >= array ) is cls  # __ge__(float)

    assert type( array +  array ) is cls  # __add__(self)
    assert type( array +  INT   ) is cls  # __add__(int)
    assert type( array +  FLOAT ) is cls  # __add__(float)
    assert type( INT   +  array ) is cls  # __radd__(int)
    assert type( FLOAT +  array ) is cls  # __radd__(float)

    assert type( array -  array ) is cls  # __sub__(self)
    assert type( array -  INT   ) is cls  # __sub__(int)
    assert type( array -  FLOAT ) is cls  # __sub__(float)
    assert type( INT   -  array ) is cls  # __rsub__(int)
    assert type( FLOAT -  array ) is cls  # __rsub__(float)

    assert type( array *  array ) is cls  # __mul__(self)
    assert type( array *  INT   ) is cls  # __mul__(int)
    assert type( array *  FLOAT ) is cls  # __mul__(float)
    assert type( INT   *  array ) is cls  # __rmul__(int)
    assert type( FLOAT *  array ) is cls  # __rmul__(float)

    assert type( array ** array ) is cls  # __pow__(self)
    assert type( array ** INT   ) is cls  # __pow__(int)
    assert type( array ** FLOAT ) is cls  # __pow__(float)
    assert type( INT   ** array ) is cls  # __rpow__(int)
    assert type( FLOAT ** array ) is cls  # __rpow__(float)

    assert type( array // array ) is cls  # __floordiv__(self)
    assert type( array // INT   ) is cls  # __floordiv__(int)
    assert type( array // FLOAT ) is cls  # __floordiv__(float)
    assert type( INT   // array ) is cls  # __rfloordiv__(int)
    assert type( FLOAT // array ) is cls  # __rfloordiv__(float)

    with pytest_xfail("arrow/#28497", strict=(case == "pandas[pa_int]")):
        assert type( array %  array ) is cls  # __mod__(self)
        assert type( array %  INT   ) is cls  # __mod__(int)
        assert type( array %  FLOAT ) is cls  # __mod__(float)
        assert type( INT   %  array ) is cls  # __rmod__(int)
        assert type( FLOAT %  array ) is cls  # __rmod__(float)

    assert type( array &  array ) is cls  # __and__(self)
    assert type( array &  INT   ) is cls  # __and__(int)
    assert type( INT   &  array ) is cls  # __rand__(int)

    assert type( array |  INT   ) is cls  # __or__(array)
    assert type( array |  INT   ) is cls  # __or__(int)
    assert type( INT   |  array ) is cls  # __ror__(int)

    assert type( array ^  array ) is cls  # __xor__(self)
    assert type( array ^  INT   ) is cls  # __xor__(int)
    assert type( INT   ^  array ) is cls  # __rxor__(int)
    # fmt: on


@pytest.mark.parametrize("case", FLOAT_ARRAYS)
def test_float_array(case: str) -> None:
    r"""Test float arrays."""
    array = FLOAT_ARRAYS[case]
    cls = type(array)
    assert_protocol(array, FloatArray)

    # test interface
    assert array.min() == -1
    assert array.max() == 1
    assert array.mean() == 0
    assert array.std() <= 1  # different results due to ddof
    assert array.var() <= 1  # different results due to ddof
    assert type(array.round()) is cls
    assert type(array.clip(-1, 1)) is cls

    # fmt: off
    assert type( abs(array) ) is cls      # __abs__
    assert type(    -array  ) is cls      # __neg__
    assert type(    +array  ) is cls      # __pos__

    assert type( array == array ) is cls  # __eq__(self)
    assert type( array == FLOAT ) is cls  # __eq__(float)
    assert type( array == INT   ) is cls  # __eq__(int)
    assert type( FLOAT == array ) is cls  # __eq__(float)
    assert type( INT   == array ) is cls  # __eq__(int)

    assert type( array != array ) is cls  # __ne__(self)
    assert type( array != FLOAT ) is cls  # __ne__(float)
    assert type( FLOAT != array ) is cls  # __ne__(float)
    assert type( array != INT   ) is cls  # __ne__(int)
    assert type( INT   != array ) is cls  # __ne__(int)

    assert type( array <  array ) is cls  # __lt__(self)
    assert type( array <  FLOAT ) is cls  # __lt__(float)
    assert type( FLOAT <  array ) is cls  # __lt__(float)
    assert type( array <  INT   ) is cls  # __lt__(int)
    assert type( INT   <  array ) is cls  # __lt__(int)

    assert type( array <= array ) is cls  # __le__(self)
    assert type( array <= FLOAT ) is cls  # __le__(float)
    assert type( FLOAT <= array ) is cls  # __le__(float)
    assert type( array <= INT   ) is cls  # __le__(int)
    assert type( INT   <= array ) is cls  # __le__(int)

    assert type( array >  array ) is cls  # __gt__(self)
    assert type( array >  FLOAT ) is cls  # __gt__(float)
    assert type( FLOAT >  array ) is cls  # __gt__(float)
    assert type( array >  INT   ) is cls  # __gt__(int)
    assert type( INT   >  array ) is cls  # __gt__(int)

    assert type( array >= array ) is cls  # __ge__(self)
    assert type( array >= FLOAT ) is cls  # __ge__(float)
    assert type( FLOAT >= array ) is cls  # __ge__(float)
    assert type( array >= INT   ) is cls  # __ge__(int)
    assert type( INT   >= array ) is cls  # __ge__(int)

    assert type( array +  array ) is cls  # __add__(self)
    assert type( array +  FLOAT ) is cls  # __add__(float)
    assert type( FLOAT +  array ) is cls  # __radd__(float)
    assert type( array +  INT   ) is cls  # __add__(int)
    assert type( INT   +  array ) is cls  # __radd__(int)

    assert type( array -  array ) is cls  # __sub__(self)
    assert type( array -  FLOAT ) is cls  # __sub__(float)
    assert type( FLOAT -  array ) is cls  # __rsub__(float)
    assert type( array -  INT   ) is cls  # __sub__(int)
    assert type( INT   -  array ) is cls  # __rsub__(int)

    assert type( array *  array ) is cls  # __mul__(self)
    assert type( array *  FLOAT ) is cls  # __mul__(float)
    assert type( FLOAT *  array ) is cls  # __rmul__(float)
    assert type( array *  INT   ) is cls  # __mul__(int)
    assert type( INT   *  array ) is cls  # __rmul__(int)

    assert type( array ** array ) is cls  # __pow__(self)
    assert type( array ** FLOAT ) is cls  # __pow__(float)
    assert type( FLOAT ** array ) is cls  # __rpow__(float)
    assert type( array ** INT   ) is cls  # __pow__(int)
    assert type( INT   ** array ) is cls  # __rpow__(int)

    assert type( array /  array ) is cls  # __truediv__(self)
    assert type( array /  FLOAT ) is cls  # __truediv__(float)
    assert type( FLOAT /  array ) is cls  # __truediv__(float))
    assert type( array /  INT   ) is cls  # __rtruediv__(int)
    assert type( INT   /  array ) is cls  # __rtruediv__(int)

    assert type( array // array ) is cls  # __floordiv__(self_
    assert type( array // FLOAT ) is cls  # __floordiv__(float)
    assert type( FLOAT // array ) is cls  # __rfloordiv__(float)
    assert type( array // INT   ) is cls  # __floordiv__(int)
    assert type( INT   // array ) is cls  # __rfloordiv__(int)

    with pytest_xfail("arrow/#28497", strict=(case == "pandas[pa_float]")):
        assert type( array %  array ) is cls  # __mod__(self)
        assert type( array %  FLOAT ) is cls  # __mod__(float)
        assert type( FLOAT %  array ) is cls  # __rmod__(float)
        assert type( array %  INT   ) is cls  # __mod__(int)
        assert type( INT   %  array ) is cls  # __rmod__(int
    # fmt: on


@pytest.mark.parametrize("case", COMPLEX_ARRAYS)
def test_complex_array(case: str) -> None:
    r"""Test complex arrays."""
    array = COMPLEX_ARRAYS[case]
    cls = type(array)

    # test interface
    assert array.sum() == 0
    assert array.mean() == 0

    with pytest_xfail(
        "pandas/#61646", strict=(case == "pandas[complex]"), defer_xfail=True
    ) as chk:
        assert array.std() <= 1  # different results due to ddof
        assert array.var() <= 1  # different results due to ddof

    # fmt: off
    assert type( abs(array) ) is cls          # __abs__
    assert type(    -array  ) is cls          # __neg__
    assert type(    +array  ) is cls          # __pos__

    assert type( array   == array   ) is cls  # __eq__(self)
    assert type( array   == COMPLEX ) is cls  # __eq__(complex)
    assert type( array   == FLOAT   ) is cls  # __eq__(float)
    assert type( array   == INT     ) is cls  # __eq__(int)
    assert type( COMPLEX == array   ) is cls  # __eq__(complex)
    assert type( FLOAT   == array   ) is cls  # __eq__(float)
    assert type( INT     == array   ) is cls  # __eq__(int)

    assert type( array   != array   ) is cls  # __ne__(self)
    assert type( array   != COMPLEX ) is cls  # __ne__(complex)
    assert type( array   != FLOAT   ) is cls  # __ne__(float)
    assert type( array   != INT     ) is cls  # __ne__(int)
    assert type( COMPLEX != array   ) is cls  # __ne__(complex)
    assert type( FLOAT   != array   ) is cls  # __ne__(float)
    assert type( INT     != array   ) is cls  # __ne__(int)

    assert type( array   +  array   ) is cls  # __add__(self)
    assert type( array   +  COMPLEX ) is cls  # __add__(complex)
    assert type( array   +  FLOAT   ) is cls  # __add__(float)
    assert type( array   +  INT     ) is cls  # __add__(int)
    assert type( COMPLEX +  array   ) is cls  # __radd__(complex)
    assert type( FLOAT   +  array   ) is cls  # __radd__(float)
    assert type( INT     +  array   ) is cls  # __radd__(int)

    assert type( array   -  array   ) is cls  # __sub__(self)
    assert type( array   -  COMPLEX ) is cls  # __sub__(complex)
    assert type( array   -  FLOAT   ) is cls  # __sub__(float)
    assert type( array   -  INT     ) is cls  # __sub__(int)
    assert type( COMPLEX -  array   ) is cls  # __rsub__(complex)
    assert type( FLOAT   -  array   ) is cls  # __rsub__(float)
    assert type( INT     -  array   ) is cls  # __rsub__(int)

    assert type( array   *  array   ) is cls  # __mul__(self)
    assert type( array   *  COMPLEX ) is cls  # __mul__(complex)
    assert type( array   *  FLOAT   ) is cls  # __mul__(float)
    assert type( array   *  INT     ) is cls  # __mul__(int)
    assert type( COMPLEX *  array   ) is cls  # __rmul__(complex)
    assert type( FLOAT   *  array   ) is cls  # __rmul__(float)
    assert type( INT     *  array   ) is cls  # __rmul__(int)

    assert type( array   /  array   ) is cls  # __truediv__(self)
    assert type( array   /  COMPLEX ) is cls  # __truediv__(complex)
    assert type( array   /  FLOAT   ) is cls  # __truediv__(float)
    assert type( array   /  INT     ) is cls  # __truediv__(int)
    assert type( COMPLEX /  array   ) is cls  # __rtruediv__(complex)
    assert type( FLOAT   /  array   ) is cls  # __rtruediv__(float)
    assert type( INT     /  array   ) is cls  # __rtruediv__(int)

    assert type( array   ** array   ) is cls  # __pow__(self)
    assert type( array   ** COMPLEX ) is cls  # __pow__(complex)
    assert type( array   ** FLOAT   ) is cls  # __pow__(float)
    assert type( array   ** INT     ) is cls  # __pow__(int)
    assert type( COMPLEX ** array   ) is cls  # __rpow__(complex)
    assert type( FLOAT   ** array   ) is cls  # __rpow__(float)
    assert type( INT     ** array   ) is cls  # __rpow__(int)
    # fmt: on

    pytest_xfail.any_failed(chk)


@pytest.mark.parametrize("case", TIME_ARRAYS)
def test_timedelta_array(case: str) -> None:
    r"""Test timedelta arrays."""
    array = TIME_ARRAYS[case]
    cls = type(array)

    # fmt: off
    assert type( abs(array) ) is cls  # __abs__(self)
    assert type(    -array  ) is cls  # __neg__(self)
    assert type(    +array  ) is cls  # __pos__(self)

    assert type( array == array ) is cls  # __eq__(self)
    assert type( array != array ) is cls  # __ne__(self)
    assert type( array <  array ) is cls  # __lt__(self)
    assert type( array <= array ) is cls  # __le__(self)
    assert type( array >  array ) is cls  # __gt__(self)
    assert type( array >= array ) is cls  # __ge__(self)

    assert type( array +  array ) is cls  # __add__(self)
    assert type( array -  array ) is cls  # __sub__(self)

    # test scalar operations (int)
    assert type( array *  INT   ) is cls  # __mul__(int)
    assert type( INT   *  array ) is cls  # __rmul__(int)
    # assert type( array /  INT   ) is cls  # __truediv__(int)
    # assert type( array //  INT  ) is cls  # __floordiv__(int)

    with pytest_xfail("numpy/#29201", strict=(case == "numpy[time]")):
        assert type( array == TIMEDELTA ) is cls  # __eq__(timedelta)
        assert type( array != TIMEDELTA ) is cls  # __ne__(timedelta)
        assert type( array <  TIMEDELTA ) is cls  # __lt__(timedelta)
        assert type( array <= TIMEDELTA ) is cls  # __le__(timedelta)
        assert type( array >  TIMEDELTA ) is cls  # __gt__(timedelta)
        assert type( array >= TIMEDELTA ) is cls  # __ge__(timedelta)

        assert type( TIMEDELTA == array ) is cls  # __eq__(timedelta)
        assert type( TIMEDELTA != array ) is cls  # __ne__(timedelta)
        assert type( TIMEDELTA <  array ) is cls  # __lt__(timedelta)
        assert type( TIMEDELTA <= array ) is cls  # __le__(timedelta)
        assert type( TIMEDELTA >  array ) is cls  # __gt__(timedelta)
        assert type( TIMEDELTA >= array ) is cls  # __ge__(timedelta)

        assert type( array     +  TIMEDELTA ) is cls  # __add__(timedelta)
        assert type( TIMEDELTA +  array     ) is cls  # __radd__(timedelta)
        assert type( array     -  TIMEDELTA ) is cls  # __sub__(timedelta)
        assert type( TIMEDELTA -  array     ) is cls  # __rsub__(timedelta)
    # fmt: on


@pytest.mark.parametrize("case", DATE_ARRAYS)
def test_datetime_array(case: str) -> None:
    r"""Test datetime arrays."""
    array = DATE_ARRAYS[case]
    cls = type(array)
    ZERO = array - array

    # fmt: off
    assert type( array == array ) is cls  # __eq__(self)
    assert type( array != array ) is cls  # __ne__(self)
    assert type( array <  array ) is cls  # __lt__(self)
    assert type( array <= array ) is cls  # __le__(self)
    assert type( array >  array ) is cls  # __gt__(self)
    assert type( array >= array ) is cls  # __ge__(self)

    assert type( array +  ZERO  ) is cls  # __add__(DurationArray)
    assert type( ZERO  +  array ) is cls  # __radd__(DurationArray)
    assert type( array -  array ) is cls  # __sub__(self)

    with pytest_xfail("numpy/#29201", strict=(case == "numpy[date]")):
        # test scalar operations (datetime)
        assert type( array == DATETIME  ) is cls  # __eq__(datetime)
        assert type( array != DATETIME  ) is cls  # __ne__(datetime)
        assert type( array <  DATETIME  ) is cls  # __lt__(datetime)
        assert type( array <= DATETIME  ) is cls  # __le__(datetime)
        assert type( array >  DATETIME  ) is cls  # __gt__(datetime)
        assert type( array >= DATETIME  ) is cls  # __ge__(datetime)

        assert type( array -  DATETIME  ) is cls  # __sub__(datetime)
        assert type( DATETIME  -  array ) is cls  # __rsub__(datetime)

        assert type( DATETIME  == array ) is cls  # __eq__(datetime)
        assert type( DATETIME  != array ) is cls  # __ne__(datetime)
        assert type( DATETIME  <  array ) is cls  # __lt__(datetime)
        assert type( DATETIME  <= array ) is cls  # __le__(datetime)
        assert type( DATETIME  >  array ) is cls  # __gt__(datetime)
        assert type( DATETIME  >= array ) is cls  # __ge__(datetime)

        assert type( array +  TIMEDELTA ) is cls  # __add__(timedelta)
        assert type( TIMEDELTA +  array ) is cls  # __radd__(timedelta)
    # fmt: on


@pytest.mark.parametrize("case", FLOAT_ARRAYS)
def test_generic_normalize(case: str) -> None:
    r"""Test normalization of numerical arrays."""
    array = FLOAT_ARRAYS[case]
    cls = type(array)

    def normalize[Arr: FloatArray](x: Arr) -> Arr:
        return (x - x.min()) / (x.max() - x.min())

    assert type(normalize(array)) is cls


def type_float_array_assignable() -> None:
    # fmt: off
    _numpy_float     : FloatArray = np.array([1.0], dtype=np.float64)
    _pandas_np_float : FloatArray = pd.Series([1.0], dtype=np.float64)
    _pandas_pa_float : FloatArray = pd.Series([1.0], dtype=_pa_float64)
    # FIXME: https://github.com/pola-rs/polars/issues/23132
    _polars_float    : FloatArray = pl.Series([1.0], dtype=pl.Float64())
    _torch_float     : FloatArray = pt.tensor([1.0], dtype=pt.float64)
    # fmt: on


def type_bool_array_assignable() -> None:
    # fmt: off
    _numpy_bool     : BooleanArray = np.array([True], dtype=np.bool_)
    _pandas_np_bool : BooleanArray = pd.Series([True], dtype=bool)
    _pandas_pa_bool : BooleanArray = pd.Series([True], dtype=_pa_bool)
    _polars_bool    : BooleanArray = pl.Series([True], dtype=pl.Boolean())
    _torch_bool     : BooleanArray = pt.tensor([True], dtype=pt.bool)
    # fmt: on


def type_int_array_assignable() -> None:
    # fmt: off
    _numpy_int     : IntegerArray = np.array([1], dtype=np.int64)
    _pandas_np_int : IntegerArray = pd.Series([1], dtype=np.int64)
    _pandas_pa_int : IntegerArray = pd.Series([1], dtype=_pa_int64)
    # FIXME: https://github.com/pola-rs/polars/issues/23132
    _polars_int    : IntegerArray = pl.Series([1], dtype=pl.Int64())
    _torch_int     : IntegerArray = pt.tensor([1], dtype=pt.int64)
    # fmt: on


def type_complex_array_assignable() -> None:
    # fmt: off
    _numpy_complex     : ComplexArray = np.array([1 + 1j], dtype=np.complex128)
    _torch_complex     : ComplexArray = pt.tensor([1 + 1j], dtype=pt.complex128)
    _pandas_np_complex : ComplexArray = pd.Series([1 + 1j], dtype=np.complex128)
    # fmt: on


def type_timedelta_array_assignable() -> None:
    # fmt: off
    _numpy_time     : SpanLikeArray = np.array([py_timedelta(days=1)], dtype="timedelta64[ns]")
    _pandas_np_time : SpanLikeArray = pd.Series([py_timedelta(days=1)], dtype="timedelta64[ns]")
    _pandas_pa_time : SpanLikeArray = pd.Series([py_timedelta(days=1)], dtype=_pa_duration_ns)
    _polars_time    : SpanLikeArray = pl.Series([py_timedelta(days=1)], dtype=pl.Time())
    # fmt: on


def type_datetime_array_assignable() -> None:
    # fmt: off
    _numpy_date     : TimeLikeArray = np.array([py_datetime(2021, 1, 1)], dtype="datetime64[ns]")
    _pandas_np_date : TimeLikeArray = pd.Series([py_datetime(2021, 1, 1)], dtype="datetime64[ns]")
    _pandas_pa_date : TimeLikeArray = pd.Series([py_datetime(2021, 1, 1)], dtype=_pa_timestamp_ns)
    _polars_date    : TimeLikeArray = pl.Series([py_datetime(2021, 1, 1)], dtype=pl.Date())
    # fmt: on
