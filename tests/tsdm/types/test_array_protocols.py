r"""Test the Array protocol."""

import logging
from array import array as python_array
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch
from typing_extensions import get_protocol_members

from tsdm.testing import assert_protocol, check_shared_interface
from tsdm.types.arrays import (
    ArrayKind,
    MutableArray,
    NumericalArray,
    NumericalSeries,
    NumericalTensor,
    SeriesKind,
    SupportsArithmetic,
    SupportsArray,
    SupportsArrayUfunc,
    SupportsComparison,
    SupportsDataFrame,
    SupportsDevice,
    SupportsDtype,
    SupportsInplaceArithmetic,
    SupportsItem,
    SupportsMatmul,
    SupportsNdim,
    SupportsShape,
    TableKind,
)

__logger__ = logging.getLogger(__name__)
RNG = np.random.default_rng()
ARRAY_PROTOCOLS = (ArrayKind, NumericalArray, MutableArray)

_BOOL_LIST = [True, False, True, False]
_STRING_LIST = ["a", "b", "c", "d"]
_INT_LIST = [1, 2, 3, 4]
_INT_MATRIX = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
_TABLE_DATA_FLOAT = {
    "x": [1.1, 2.2, 3.3, 4.4],
    "y": [5.5, 6.6, 7.7, 8.8],
    "z": [9.9, 0.0, 1.1, 2.2],
}
_TABLE_DATA_MIXED = {
    "label": ["a", "b", "c", "d"],
    "x": [1.1, 2.2, 3.3, 4.4],
    "y": [5.5, 6.6, 7.7, 8.8],
}
_DATETIME_LIST = pd.date_range("2021-01-01", periods=4)

# tensorial (single dtype)
NP_ARRAY_1D = np.array(_INT_LIST)
NP_ARRAY_2D = np.array(_INT_MATRIX)
PA_ARRAY_INT = pa.array(_INT_LIST)
PA_ARRAY_STR = pa.array(_STRING_LIST)
PD_INDEX_STR = pd.Index(_STRING_LIST)
PD_INDEX_INT = pd.Index(_INT_LIST)
PD_MULTIINDEX = pd.MultiIndex.from_tuples([("a", 1), ("b", 2), ("c", 3)])
PD_SERIES_INT = pd.Series(_INT_LIST, index=_DATETIME_LIST)
PD_SERIES_STR = pd.Series(_STRING_LIST, index=_DATETIME_LIST)
PD_ARRAY_INT = pd.Series(_INT_LIST).array
PD_ARRAY_STR = pd.Series(_STRING_LIST).array
PD_ARRAY_PD_INT = pd.Series(_INT_LIST, dtype="Int64").array
PD_ARRAY_PD_STR = pd.Series(_STRING_LIST, dtype="string").array
PD_ARRAY_PA_INT = pd.Series(_INT_LIST, dtype="int64[pyarrow]").array
PD_ARRAY_PA_STR = pd.Series(_INT_LIST, dtype="string[pyarrow]").array
PL_SERIES_INT = pl.Series(_INT_LIST)
PL_SERIES_STR = pl.Series(_STRING_LIST)
PT_TENSOR_1D = torch.tensor(_INT_LIST)
PT_TENSOR_2D = torch.tensor(_INT_MATRIX)
PY_ARRAY = memoryview(python_array("i", [1, 2, 3]))

# tabular (mixed dtype)
PA_TABLE_FLOAT = pa.table(_TABLE_DATA_FLOAT)
PA_TABLE_MIXED = pa.table(_TABLE_DATA_MIXED)
PD_TABLE_FLOAT = pd.DataFrame(_TABLE_DATA_FLOAT, index=_DATETIME_LIST)
PD_TABLE_MIXED = pd.DataFrame(_TABLE_DATA_MIXED, index=_DATETIME_LIST)
PL_TABLE_FLOAT = pl.DataFrame(_TABLE_DATA_FLOAT)
PL_TABLE_MIXED = pl.DataFrame(_TABLE_DATA_MIXED)


TEST_ARRAYS = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_multiindex"   : PD_MULTIINDEX,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "pandas_table_mixed"  : PD_TABLE_MIXED,
    "polars_series_int"   : PL_SERIES_INT,
    "polars_series_str"   : PL_SERIES_STR,
    "polars_table_float"  : PL_TABLE_FLOAT,
    "polars_table_mixed"  : PL_TABLE_MIXED,
    "pyarrow_array_int"   : PA_ARRAY_INT,
    "pyarrow_array_str"   : PA_ARRAY_STR,
    "pyarrow_table_float" : PA_TABLE_FLOAT,
    "pyarrow_table_mixed" : PA_TABLE_MIXED,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip
r"""Collection of all test arrays."""

SUPPORTS_ARRAYS: dict[str, SupportsArray] = TEST_ARRAYS.copy()
r"""Collection of all test arrays that satisfy the `SupportsArray` protocol."""

SUPPORTS_DTYPE: dict[str, SupportsDtype] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "polars_series_int"   : PL_SERIES_INT,
    "polars_series_str"   : PL_SERIES_STR,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip

SUPPORTS_ARRAYS_UFUNC: dict[str, SupportsArrayUfunc] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "polars_series_int"   : PL_SERIES_INT,
}  # fmt: skip

SERIES: dict[str, SeriesKind[str]] = {
    "pandas_array_int"  : PD_ARRAY_INT,
    "pandas_array_str"  : PD_ARRAY_STR,
    "pandas_index_int"  : PD_INDEX_INT,
    "pandas_index_str"  : PD_INDEX_STR,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "polars_series_int" : PL_SERIES_INT,
    "polars_series_str" : PL_SERIES_STR,
    "pyarrow_array_int" : PA_ARRAY_INT,
    "pyarrow_array_str" : PA_ARRAY_STR,
}  # fmt: skip

TABLES: dict[str, TableKind] = {
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "pandas_table_mixed"  : PD_TABLE_MIXED,
    "polars_table_float"  : PL_TABLE_FLOAT,
    "polars_table_mixed"  : PL_TABLE_MIXED,
    "pyarrow_table_float" : PA_TABLE_FLOAT,
    "pyarrow_table_mixed" : PA_TABLE_MIXED,
}  # fmt: skip

ARRAYS: dict[str, ArrayKind] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "pandas_table_mixed"  : PD_TABLE_MIXED,
    "polars_series_int"   : PL_SERIES_INT,
    "polars_series_str"   : PL_SERIES_STR,
    "polars_table_float"  : PL_TABLE_FLOAT,
    "polars_table_mixed"  : PL_TABLE_MIXED,
    "pyarrow_table_float" : PA_TABLE_FLOAT,
    "pyarrow_table_mixed" : PA_TABLE_MIXED,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip

NUMERICAL_ARRAYS: dict[str, NumericalArray] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "pandas_table_mixed"  : PD_TABLE_MIXED,
    "polars_series_int"   : PL_SERIES_INT,
    "polars_series_str"   : PL_SERIES_STR,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip

NUMERICAL_SERIES: dict[str, NumericalSeries] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "polars_series_int"   : PL_SERIES_INT,
    "polars_series_str"   : PL_SERIES_STR,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip

NUMERICAL_TENSORS: dict[str, NumericalTensor] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_array_int"    : PD_ARRAY_INT,
    "pandas_array_str"    : PD_ARRAY_STR,
    "pandas_index_int"    : PD_INDEX_INT,
    "pandas_index_str"    : PD_INDEX_STR,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip

MUTABLE_ARRAYS: dict[str, MutableArray] = {
    "numpy_ndarray_1d"    : NP_ARRAY_1D,
    "numpy_ndarray_2d"    : NP_ARRAY_2D,
    "pandas_series_int"   : PD_SERIES_INT,
    "pandas_series_str"   : PD_SERIES_STR,
    "pandas_table_float"  : PD_TABLE_FLOAT,
    "pandas_table_mixed"  : PD_TABLE_MIXED,
    "torch_tensor_1d"     : PT_TENSOR_1D,
    "torch_tensor_2d"     : PT_TENSOR_2D,
}  # fmt: skip

EXAMPLES_BY_PROTOCOL: dict[type, dict[str, Any]] = {
    SeriesKind      : SERIES,
    TableKind       : TABLES,
    ArrayKind       : ARRAYS,
    NumericalArray  : NUMERICAL_ARRAYS,
    NumericalSeries : NUMERICAL_SERIES,
    NumericalTensor : NUMERICAL_TENSORS,
    MutableArray    : MUTABLE_ARRAYS,
}  # fmt: skip
r"""Examples by protocol."""

DUNDER_ARITHMETIC: frozenset[str] = frozenset({
    # comparisons
    "__eq__",
    "__ge__",
    "__gt__",
    "__le__",
    "__lt__",
    "__ne__",
    # unary ops
    "__hash__",
    "__invert__",
    "__abs__",
    "__pos__",
    "__neg__",
    # binary ops
    "__add__",
    "__and__",
    "__divmod__",
    "__floordiv__",
    "__lshift__",
    "__matmul__",
    "__mod__",
    "__mul__",
    "__or__",
    "__pow__",
    "__sub__",
    "__truediv__",
    "__xor__",
    # inplace ops
    "__iadd__",
    "__iand__",
    "__ifloordiv__",
    "__ilshift__",
    "__imatmul__",
    "__imod__",
    "__imul__",
    "__ior__",
    "__ipow__",
    "__irshift__",
    "__isub__",
    "__itruediv__",
    "__ixor__",
    # r-ops
    "__radd__",
    "__rand__",
    "__rdivmod__",
    "__rfloordiv__",
    "__rlshift__",
    "__rmatmul__",
    "__rmod__",
    "__rmul__",
    "__ror__",
    "__rpow__",
    "__rrshift__",
    "__rshift__",
    "__rsub__",
    "__rtruediv__",
    "__rxor__",
})
r"""Dunder methods for arithmetic operations."""

EXCLUDED_MEMBERS: dict[type, set[str]] = {
    ArrayKind       : set(),
    SeriesKind      : {"to_numpy"},
    TableKind       : {"columns", "join", "drop", "filter"},
    NumericalArray  : set(),
    NumericalSeries : set(),
    NumericalTensor : {
        "T", "transpose",
        "argsort", "nbytes", "repeat",
        "size", "take", "view",
    },
    MutableArray    : {
        "T", "transpose",
        "clip", "cumprod", "cumsum", "dot",
        "mean", "ndim", "prod",
        "size", "squeeze", "std", "sum",
        "swapaxes", "var",
    },
}  # fmt: skip
r"""Excluded members for each protocol."""


def is_admissable(name: str) -> bool:
    r"""Check if the name is admissable."""
    return name in DUNDER_ARITHMETIC or not name.startswith("_")


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_array(name: str) -> None:
    r"""Test the SupportsArray protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsArray)
    assert issubclass(obj.__class__, SupportsArray)
    assert isinstance(obj.__array__(), np.ndarray)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_len(name: str) -> None:
    r"""Test the SupportsLen protocol."""
    obj = TEST_ARRAYS[name]
    assert hasattr(obj, "__len__")
    result = len(obj)
    assert isinstance(result, int)
    assert result == 4


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_array_ufunc(name: str) -> None:
    r"""Test the SupportsArrayUfunc protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsArrayUfunc)
    assert issubclass(obj.__class__, SupportsArrayUfunc)

    # test ufunc
    result = np.exp(obj)
    assert isinstance(result, type(obj))


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_dataframe(name: str) -> None:
    r"""Test the SupportsDataFrame protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsDataFrame)
    assert isinstance(obj.__dataframe__(), TableKind)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_dtype(name: str) -> None:
    r"""Test the SupportsDtype protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsDtype)
    assert isinstance(obj.dtype, object)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_shape(name: str) -> None:
    r"""Test the SupportsShape protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsShape)
    assert isinstance(obj.shape, tuple)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_ndim(name: str) -> None:
    r"""Test the SupportsNdim protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsNdim)
    assert isinstance(obj.ndim, int)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_device(name: str) -> None:
    r"""Test the SupportsDevice protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsDevice)
    assert isinstance(obj.device, object)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_matmul(name: str) -> None:
    r"""Test the SupportsMatmul protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsMatmul)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_item(name: str) -> None:
    r"""Test the SupportsShape protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsItem)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_comparison(name: str) -> None:
    r"""Test the SupportsComparison protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsComparison)
    try:
        _ = obj < obj
    except TypeError as exc:
        raise AssertionError(f"Comparison failed for {name}!") from exc


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_arithmetic(name: str) -> None:
    r"""Test the SupportsArithmetic protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsArithmetic)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_inplace(name: str) -> None:
    r"""Test the SupportsInplaceArithmetic protocol."""
    obj = TEST_ARRAYS[name]
    assert_protocol(obj, SupportsInplaceArithmetic)


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_itering(name: str) -> None:
    r"""Test if the object supports iteration."""
    obj = TEST_ARRAYS[name]
    try:
        next(iter(obj))
    except Exception:
        raise AssertionError(f"Failed to iterate over {name}!") from None


@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_supports_getitem_int(name: str) -> None:
    r"""Test if the object supports integer indexing."""
    obj = TEST_ARRAYS[name]
    try:
        obj[0]
    except Exception:
        raise AssertionError(f"Failed to index {name}!") from None


@pytest.mark.parametrize("name", SERIES)
def test_series(name: str) -> None:
    r"""Test the Series protocol."""
    series = SERIES[name]
    cls = series.__class__
    scalar_type = int | str | np.generic | pa.Scalar

    assert isinstance(series, SeriesKind)

    attrs = set(get_protocol_members(SeriesKind)) - DUNDER_ARITHMETIC

    assert isinstance(series.__array__(), np.ndarray)
    attrs.remove("__array__")

    assert isinstance(len(series), int)
    attrs.remove("__len__")

    for x in series:
        assert isinstance(x, scalar_type)
    attrs.remove("__iter__")

    assert isinstance(series[0], scalar_type)
    assert isinstance(series[0:2], cls)
    attrs.remove("__getitem__")

    assert series.equals(series)
    attrs.remove("equals")

    # assert isinstance(series.unique(), cls | np.ndarray)
    # attrs.remove("unique")

    # value_counts = series.value_counts()
    # assert isinstance(value_counts, SupportsArray)
    # attrs.remove("value_counts")

    # check that all attributes are tested
    assert not attrs, f"Forgot to test: {attrs}!"


@pytest.mark.parametrize("name", TABLES)
def test_table(name: str) -> None:
    r"""Test the Table protocol."""
    table = TABLES[name]
    assert isinstance(table, TableKind)
    # assert not isinstance(table, SeriesKind)

    # check methods
    attrs = set(get_protocol_members(TableKind)) - DUNDER_ARITHMETIC

    assert isinstance(table.__array__(), np.ndarray)
    attrs.remove("__array__")

    table.__dataframe__()
    attrs.remove("__dataframe__")

    assert isinstance(len(table), int)
    attrs.remove("__len__")

    assert isinstance(table.shape, tuple)
    assert len(table.shape) == 2
    assert isinstance(table.shape[0], int)
    assert isinstance(table.shape[1], int)
    attrs.remove("shape")

    assert isinstance(table["x"], SeriesKind)
    attrs.remove("__getitem__")

    assert table.equals(table)
    attrs.remove("equals")

    # check that all attributes are tested
    assert not attrs, f"Forgot to test: {attrs}!"


@pytest.mark.parametrize("name", ARRAYS)
def test_array(name: str) -> None:
    r"""Test the Array protocol."""
    array = ARRAYS[name]
    assert_protocol(array, ArrayKind)


@pytest.mark.parametrize("name", NUMERICAL_ARRAYS)
def test_numerical_array(name: str) -> None:
    r"""Test the NumericalArray protocol."""
    numerical_array = NUMERICAL_ARRAYS[name]
    assert_protocol(numerical_array, NumericalArray)


@pytest.mark.parametrize("name", NUMERICAL_TENSORS)
def test_numerical_tensor_getitem(name: str) -> None:
    tensor = NUMERICAL_TENSORS[name]
    cls = type(tensor)
    scalar_type = type(tensor.ravel()[0])

    # ensure length, otherwise other checks can fail.
    assert len(tensor) == 4

    # int
    assert isinstance(tensor[0], scalar_type | cls)
    # slice
    assert isinstance(tensor[:-1], cls)
    # range
    assert isinstance(tensor[range(2)], cls)
    # list[int]
    assert isinstance(tensor[[0, 1]], cls)
    # list[bool]
    assert isinstance(tensor[[True, False, True, False]], cls)
    # Ellipsis
    assert isinstance(tensor[...], cls)

    if len(tensor.shape) > 1:
        assert tensor.shape[1] == 3
        # tuple[int, int]
        assert isinstance(tensor[0, 0], scalar_type | cls)
        # tuple[slice, slice]
        assert isinstance(tensor[0:1, 0:1], cls)
        # tuple[slice, list[bool]]
        assert isinstance(tensor[:, [True, False, True]], cls)
        # tuple[range, range]
        assert isinstance(tensor[range(1), range(1)], cls)
        # tuple[list[int], list[int]]
        assert isinstance(tensor[[0, 1], [0]], cls)
        # tuple[int, Ellipsis]
        assert isinstance(tensor[0, ...], cls)
        # tuple[Ellipsis, int]
        assert isinstance(tensor[..., 0], cls)
        # tuple[Ellipsis, slice]
        assert isinstance(tensor[..., 0:1], cls)
        # tuple[Ellipsis, range]
        assert isinstance(tensor[..., range(1)], cls)
        # tuple[Ellipsis, list[int]]
        assert isinstance(tensor[..., [0, 1]], cls)


@pytest.mark.parametrize("name", MUTABLE_ARRAYS)
def test_mutable_array(name: str) -> None:
    r"""Test the MutableArray protocol."""
    mutable_array = MUTABLE_ARRAYS[name]
    assert_protocol(mutable_array, MutableArray)


@pytest.mark.parametrize("proto", EXAMPLES_BY_PROTOCOL)
@pytest.mark.parametrize("name", TEST_ARRAYS)
def test_all_protocols(proto: type, name: str) -> None:
    r"""Test the NumericalArray protocol."""
    obj = TEST_ARRAYS[name]
    if name in EXAMPLES_BY_PROTOCOL[proto]:
        assert_protocol(obj, proto)
    else:
        with pytest.raises(AssertionError):
            assert_protocol(obj, proto)


@pytest.mark.parametrize("protocol", EXAMPLES_BY_PROTOCOL)
def test_shared_attrs(protocol: type) -> None:
    r"""Test which shared attributes exist that are not covered by protocols."""
    examples = EXAMPLES_BY_PROTOCOL[protocol]
    check_shared_interface(examples.values(), protocol, raise_on_extra=False)


def test_table_manual() -> None:
    r"""Test the Table protocol (shape and __len__ and __getitem__)."""
    LOGGER = __logger__.getChild(SupportsShape.__name__)
    LOGGER.info("Testing.")

    torch_tensor: torch.Tensor = torch.tensor([1, 2, 3])
    torch_table: SupportsShape = torch_tensor
    assert isinstance(
        torch_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(torch_table))}"

    numpy_ndarray: np.ndarray = np.array([1, 2, 3])
    numpy_table: SupportsShape = numpy_ndarray
    assert isinstance(
        numpy_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(numpy_table))}"

    pandas_frame: pd.DataFrame = pd.DataFrame(RNG.normal(size=(3, 3)))
    pandas_table: SupportsShape = pandas_frame
    assert isinstance(
        pandas_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_table))}"

    pandas_series: pd.Series = pd.Series([1, 2, 3])
    pandas_series_array: SupportsShape = pandas_series
    assert isinstance(
        pandas_series_array, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_series_array))}"

    pandas_index: pd.Index = pd.Index([1, 2, 3])
    pandas_index_array: SupportsShape = pandas_index
    assert isinstance(
        pandas_index_array, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_index_array))}"

    pyarrow_frame: pa.Table = pa.Table.from_pandas(pandas_frame)
    pyarrow_table: SupportsShape = pyarrow_frame
    assert isinstance(
        pyarrow_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pyarrow_table))}"

    pyarrow_series: pa.Array = pa.Array.from_pandas(pandas_series)
    pyarrow_series_table: SupportsShape = pyarrow_series
    assert isinstance(
        pyarrow_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pyarrow_series_table))}"

    tables = [
        torch_table,
        numpy_table,
        pandas_table,
        pandas_series_array,
        pandas_index_array,
        pyarrow_table,
        pyarrow_series_table,
    ]
    shared_attrs = set.intersection(*(set(dir(tab)) for tab in tables))
    __logger__.info("\nShared members of Tables: %s", shared_attrs)
