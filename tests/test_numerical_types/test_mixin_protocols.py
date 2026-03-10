r"""Test other protocols."""

import numpy as np
import pytest

from numerical_types import (
    SupportsArray,
    SupportsArrayUfunc,
    SupportsComparison,
    SupportsDataFrame,
    SupportsDevice,
    SupportsDtype,
    SupportsItem,
    SupportsMatmul,
    SupportsNdim,
    SupportsShape,
)
from tsdm.testing import assert_protocol

from .fixtures import ARRAYS1D, ARRAYS2D, SERIES, TABLES

TEST_ARRAYS = {
    # "pandas_series_noindex_int": PD_SERIES_NOINDEX_INT,
    # "pandas_series_noindex_str": PD_SERIES_NOINDEX_STR,
    # "pandas_table_noindex_float": PD_TABLE_NOINDEX_FLOAT,
    # "pandas_table_noindex_mixed": PD_TABLE_NOINDEX_MIXED,
    # "pandas_array_int"    : PD_ARRAY_INT,
    # "pandas_array_str"    : PD_ARRAY_STR,
    # "pandas_index_int"    : PD_INDEX_INT,
    # "pandas_index_str"    : PD_INDEX_STR,
    # "pandas_multiindex"   : PD_MULTIINDEX,
    # "pyarrow_array_int"   : PA_ARRAY_INT,
    # "pyarrow_array_str"   : PA_ARRAY_STR,

    "numpy_ndarray_1d"    : ARRAYS1D.NP.INT,
    "numpy_ndarray_2d"    : ARRAYS2D.NP.INT,
    "torch_tensor_1d"     : ARRAYS1D.PT.FLOAT,
    "torch_tensor_2d"     : ARRAYS2D.PT.FLOAT,

    "pandas_series_int"   : SERIES.PD_NP.INT,
    "pandas_series_str"   : SERIES.PD_NP.STRING,
    "polars_series_int"   : SERIES.PL.INT,
    "polars_series_str"   : SERIES.PL.STRING,

    "table_pandas_float"  : TABLES.PD_NP.FLOAT,
    "table_pandas_mixed"  : TABLES.PD_NP.MIXED,
    "table_polars_float"  : TABLES.PL.FLOAT,
    "table_polars_mixed"  : TABLES.PL.MIXED,
    "table_pyarrow_float" : TABLES.PA.FLOAT,
    "table_pyarrow_mixed" : TABLES.PA.MIXED,
}  # fmt: skip
r"""Collection of all test arrays."""


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_array(name: str) -> None:
    r"""Test the SupportsArray protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsArray)
    assert issubclass(obj.__class__, SupportsArray)
    assert isinstance(obj.__array__(), np.ndarray)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_len(name: str) -> None:
    r"""Test the SupportsLen protocol."""
    obj = TEST_ARRAYS[name]
    assert hasattr(obj, "__len__")
    result = len(obj)
    assert isinstance(result, int)
    assert result == 4


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_array_ufunc(name: str) -> None:
    r"""Test the SupportsArrayUfunc protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsArrayUfunc)
    assert issubclass(obj.__class__, SupportsArrayUfunc)

    result = np.exp(obj)
    assert isinstance(result, type(obj))


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_dataframe(name: str) -> None:
    r"""Test the SupportsDataFrame protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsDataFrame)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_dtype(name: str) -> None:
    r"""Test the SupportsDtype protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsDtype)
    assert isinstance(obj.dtype, object)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_shape(name: str) -> None:
    r"""Test the SupportsShape protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsShape)
    assert isinstance(obj.shape, tuple)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_ndim(name: str) -> None:
    r"""Test the SupportsNdim protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsNdim)
    assert isinstance(obj.ndim, int)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_device(name: str) -> None:
    r"""Test the SupportsDevice protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsDevice)
    assert isinstance(obj.device, object)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_matmul(name: str) -> None:
    r"""Test the SupportsMatmul protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsMatmul)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_item(name: str) -> None:
    r"""Test the SupportsShape protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsItem)


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_comparison(name: str) -> None:
    r"""Test the SupportsComparison protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsComparison)
    try:
        _ = obj < obj
    except TypeError as exc:
        raise AssertionError(f"Comparison failed for {name}!") from exc


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_itering(name: str) -> None:
    r"""Test if the object supports iteration."""
    obj = TEST_ARRAYS[name]
    try:
        next(iter(obj))
    except Exception:
        raise AssertionError(f"Failed to iterate over {name}!") from None


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_getitem_int(name: str) -> None:
    r"""Test if the object supports integer indexing."""
    obj = TEST_ARRAYS[name]
    try:
        obj[0]
    except Exception:
        raise AssertionError(f"Failed to index {name}!") from None
