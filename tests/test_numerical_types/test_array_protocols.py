r"""Test numerical arrays."""
# mypy: disable-error-code="unreachable"

from typing import TYPE_CHECKING

import pytest

from numerical_types import (
    BooleanArray,
    ComplexArray,
    FloatArray,
    IntegerArray,
    SpanLikeArray,
    TimeLikeArray,
)
from test_utils import pytest_xfail
from tsdm.testing import assert_protocol

from .fixtures import (
    BOOL,
    COMPLEX,
    DATETIME,
    FLOAT,
    INT,
    SERIES,
    TIMEDELTA,
)

BOOL_ARRAYS: dict[str, BooleanArray] = {
    "numpy[bool]"     : SERIES.NP.BOOL,
    "pandas[np_bool]" : SERIES.PD_NP.BOOL,
    "pandas[pa_bool]" : SERIES.PD_PA.BOOL,
    "polars[bool]"    : SERIES.PL.BOOL,
    "torch[bool]"     : SERIES.PT.BOOL,
}  # fmt: skip
r"""Dictionary of bool arrays."""

INT_ARRAYS: dict[str, IntegerArray] = {
    "numpy[int]"     : SERIES.NP.INT,
    "pandas[np_int]" : SERIES.PD_NP.INT,
    "pandas[pa_int]" : SERIES.PD_PA.INT,
    "polars[int]"    : SERIES.PL.INT,
    "torch[int]"     : SERIES.PT.INT,
}  # fmt: skip
r"""Dictionary of int arrays."""

FLOAT_ARRAYS: dict[str, FloatArray] = {
    "numpy[float]"     : SERIES.NP.FLOAT,
    "pandas[np_float]" : SERIES.PD_NP.FLOAT,
    "pandas[pa_float]" : SERIES.PD_PA.FLOAT,
    "polars[float]"    : SERIES.PL.FLOAT,
    "torch[float]"     : SERIES.PT.FLOAT,
}  # fmt: skip
r"""Dictionary of float arrays."""

COMPLEX_ARRAYS: dict[str, ComplexArray] = {
    "numpy[complex]"     : SERIES.NP.COMPLEX,
    "torch[complex]"     : SERIES.PT.COMPLEX,
    "pandas[np_complex]" : SERIES.PD_NP.COMPLEX,
}  # fmt: skip
r"""Dictionary of complex arrays."""

TIME_ARRAYS: dict[str, SpanLikeArray] = {
    "numpy[time]"     : SERIES.NP.TIMEDELTA,
    "pandas[np_time]" : SERIES.PD_NP.TIMEDELTA,
    "pandas[pa_time]" : SERIES.PD_PA.TIMEDELTA,
    "polars[time]"    : SERIES.PL.TIMEDELTA,
}  # fmt: skip
r"""Dictionary of timedelta arrays."""

DATE_ARRAYS: dict[str, TimeLikeArray] = {
    "numpy[date]"     : SERIES.NP.DATETIME,
    "pandas[np_date]" : SERIES.PD_NP.DATETIME,
    "pandas[pa_date]" : SERIES.PD_PA.DATETIME,
    "polars[date]"    : SERIES.PL.DATETIME,
}  # fmt: skip
r"""Dictionary of datetime arrays."""


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

    if not TYPE_CHECKING:
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
