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

from tsdm.testing import assert_protocol
from tsdm.types.protocols import (
    ArrayKind,
    MutableArray,
    NumericalArray,
    NumericalSeries,
    NumericalTensor,
    SeriesKind,
    SupportsArray,
    SupportsShape,
    TableKind,
)

__logger__ = logging.getLogger(__name__)
RNG = np.random.default_rng()
ARRAY_PROTOCOLS = (ArrayKind, NumericalArray, MutableArray)

_STRING_LIST = ["a", "b", "c"]
_INT_LIST = [1, 2, 3]
_INT_MATRIX = [[1, 2], [3, 4], [5, 6], [7, 8]]
_TABLE_DATA = {
    "int_col": [1, 2, 3, 4],
    "float_col": [1.1, 2.2, 3.3, 4.4],
    "string_col": ["a", "b", "c", "d'"],
}

# tensorial (single dtype)
NP_ARRAY_1D = np.array(_INT_LIST)
NP_ARRAY_2D = np.array(_INT_MATRIX)
PA_ARRAY_INT = pa.array(_INT_LIST)
PA_ARRAY_STR = pa.array(_STRING_LIST)
PD_INDEX_STR = pd.Index(_STRING_LIST)
PD_INDEX_INT = pd.Index(_INT_LIST)
PD_SERIES_INT = pd.Series(_INT_LIST)
PD_SERIES_STR = pd.Series(_STRING_LIST)
PL_SERIES_INT = pl.Series(_INT_LIST)
PL_SERIES_STR = pl.Series(_STRING_LIST)
PT_TENSOR_1D = torch.tensor(_INT_LIST)
PT_TENSOR_2D = torch.tensor(_INT_MATRIX)
PY_ARRAY = memoryview(python_array("i", [1, 2, 3]))

# tabular (mixed dtype)
PA_TABLE = pa.table(_TABLE_DATA)
PD_DATAFRAME = pd.DataFrame(_TABLE_DATA)
PL_DATAFRAME = pl.DataFrame(_TABLE_DATA)

TEST_OBJECTS = {
    "python_array"      : PY_ARRAY,
    "numpy_ndarray_1d"  : NP_ARRAY_1D,
    "numpy_ndarray_2d"  : NP_ARRAY_2D,
    "pandas_index_int"  : PD_INDEX_INT,
    "pandas_index_str"  : PD_INDEX_STR,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "pandas_dataframe"  : PD_DATAFRAME,
    "polars_series_str" : PL_SERIES_STR,
    "polars_series_int" : PL_SERIES_INT,
    "polars_dataframe"  : PL_DATAFRAME,
    "pyarrow_array_str" : PA_ARRAY_STR,
    "pyarrow_array_int" : PA_ARRAY_INT,
    "pyarrow_table"     : PA_TABLE,
    "torch_tensor_1d"   : PT_TENSOR_1D,
    "torch_tensor_2d"   : PT_TENSOR_2D,
}  # fmt: skip

SUPPORTS_ARRAYS: dict[str, SupportsArray] = {
    "numpy_ndarray_1d"  : NP_ARRAY_1D,
    "numpy_ndarray_2d"  : NP_ARRAY_2D,
    "pandas_dataframe"  : PD_DATAFRAME,
    "pandas_index_int"  : PD_INDEX_INT,
    "pandas_index_str"  : PD_INDEX_STR,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "polars_dataframe"  : PL_DATAFRAME,
    "polars_series_str" : PL_SERIES_STR,
    "polars_series_int" : PL_SERIES_INT,
    "pyarrow_array_str" : PA_ARRAY_STR,
    "pyarrow_array_int" : PA_ARRAY_INT,
    "pyarrow_table"     : PA_TABLE,
    "torch_tensor_1d"   : PT_TENSOR_1D,
    "torch_tensor_2d"   : PT_TENSOR_2D,
}  # fmt: skip

SERIES: dict[str, SeriesKind[str]] = {
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
    "pandas_dataframe" : PD_DATAFRAME,
    "polars_dataframe" : PL_DATAFRAME,
    "pyarrow_table"    : PA_TABLE,
}  # fmt: skip

ARRAYS: dict[str, ArrayKind] = {
    "numpy_ndarray_1d"  : NP_ARRAY_1D,
    "numpy_ndarray_2d"  : NP_ARRAY_2D,
    "pandas_dataframe"  : PD_DATAFRAME,
    "pandas_index_int"  : PD_INDEX_INT,
    "pandas_index_str"  : PD_INDEX_STR,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "polars_dataframe"  : PL_DATAFRAME,
    "polars_series_int" : PL_SERIES_INT,
    "polars_series_str" : PL_SERIES_STR,
    "pyarrow_table"     : PA_TABLE,
    "torch_tensor_1d"   : PT_TENSOR_1D,
    "torch_tensor_2d"   : PT_TENSOR_2D,
}  # fmt: skip

NUMERICAL_ARRAYS: dict[str, NumericalArray] = {
    "numpy_ndarray_1d"  : NP_ARRAY_1D,
    "numpy_ndarray_2d"  : NP_ARRAY_2D,
    "pandas_dataframe"  : PD_DATAFRAME,
    "pandas_index_int"  : PD_INDEX_INT,
    "pandas_index_str"  : PD_INDEX_STR,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "polars_series_int" : PL_SERIES_INT,
    "polars_series_str" : PL_SERIES_STR,
    "torch_tensor_1d"   : PT_TENSOR_1D,
    "torch_tensor_2d"   : PT_TENSOR_2D,
}  # fmt: skip

NUMERICAL_SERIES: dict[str, NumericalSeries] = {
    "numpy_ndarray_1d"  : NP_ARRAY_1D,
    "numpy_ndarray_2d"  : NP_ARRAY_2D,
    "pandas_index_int"  : PD_INDEX_INT,
    "pandas_index_str"  : PD_INDEX_STR,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "polars_series_int" : PL_SERIES_INT,
    "polars_series_str" : PL_SERIES_STR,
    "torch_tensor_1d"   : PT_TENSOR_1D,
    "torch_tensor_2d"   : PT_TENSOR_2D,
}  # fmt: skip

NUMERICAL_TENSORS: dict[str, NumericalTensor] = {
    "numpy_ndarray_1d"  : NP_ARRAY_1D,
    "numpy_ndarray_2d"  : NP_ARRAY_2D,
    "pandas_index_int"  : PD_INDEX_INT,
    "pandas_index_str"  : PD_INDEX_STR,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "torch_tensor_1d"   : PT_TENSOR_1D,
    "torch_tensor_2d"   : PT_TENSOR_2D,
}  # fmt: skip

MUTABLE_ARRAYS: dict[str, MutableArray] = {
    "numpy_ndarray_1d"  : NP_ARRAY_1D,
    "numpy_ndarray_2d"  : NP_ARRAY_2D,
    "pandas_dataframe"  : PD_DATAFRAME,
    "pandas_series_int" : PD_SERIES_INT,
    "pandas_series_str" : PD_SERIES_STR,
    "torch_tensor_1d"   : PT_TENSOR_1D,
    "torch_tensor_2d"   : PT_TENSOR_2D,
}  # fmt: skip

EXAMPLES: dict[type, dict[str, Any]] = {
    SeriesKind      : SERIES,
    TableKind       : TABLES,
    ArrayKind       : ARRAYS,
    NumericalArray  : NUMERICAL_ARRAYS,
    NumericalSeries : NUMERICAL_SERIES,
    NumericalTensor : NUMERICAL_TENSORS,
    MutableArray    : MUTABLE_ARRAYS,
}  # fmt: skip

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
    SeriesKind      : {"diff", "to_numpy"},
    TableKind       : {"columns", "join", "drop", "filter"},
    NumericalArray  : set(),
    NumericalSeries : set(),
    NumericalTensor : {
        "T", "transpose",
        "__iadd__", "argsort", "nbytes", "repeat",
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


@pytest.mark.parametrize("name", SUPPORTS_ARRAYS)
def test_supports_array(name: str) -> None:
    r"""Test the SupportsArray protocol."""
    obj = SUPPORTS_ARRAYS[name]
    assert isinstance(obj, SupportsArray)
    assert issubclass(obj.__class__, SupportsArray)
    assert isinstance(obj.__array__(), np.ndarray)


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

    assert isinstance(series.unique(), cls | np.ndarray)
    attrs.remove("unique")

    assert series.equals(series)
    attrs.remove("equals")

    value_counts = series.value_counts()
    assert isinstance(value_counts, SupportsArray)
    attrs.remove("value_counts")

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

    assert isinstance(table["float_col"], SeriesKind)
    assert isinstance(table[:], TableKind)

    assert table.equals(table)
    attrs.remove("equals")

    assert isinstance(table["int_col"], SeriesKind)
    attrs.remove("__getitem__")

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
    assert len(tensor) >= 3

    # int
    assert isinstance(tensor[0], scalar_type | cls)
    # slice
    assert isinstance(tensor[:-1], cls)
    # range
    assert isinstance(tensor[range(2)], cls)
    # list[int]
    assert isinstance(tensor[[0, 1]], cls)
    # Ellipsis
    assert isinstance(tensor[...], cls)

    if len(tensor.shape) > 1:
        # tuple[int, int]
        assert isinstance(tensor[0, 0], scalar_type | cls)
        # tuple[slice, slice]
        assert isinstance(tensor[0:1, 0:1], cls)
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


@pytest.mark.parametrize("proto", EXAMPLES)
@pytest.mark.parametrize("name", TEST_OBJECTS)
def test_all_protocols(proto: type, name: str) -> None:
    r"""Test the NumericalArray protocol."""
    obj = TEST_OBJECTS[name]
    if name in EXAMPLES[proto]:
        assert_protocol(obj, proto)
    else:
        with pytest.raises(AssertionError):
            assert_protocol(obj, proto)


@pytest.mark.parametrize("proto", EXAMPLES)
def test_shared_attrs(proto: type) -> None:
    r"""Test which shared attributes exist that are not covered by protocols."""
    print("\nShared Attributes not covered by protocols:")
    examples = EXAMPLES[proto]
    protocol_members = get_protocol_members(proto)
    shared_attrs = set.intersection(*(set(dir(s)) for s in examples.values()))

    excluded_attrs = EXCLUDED_MEMBERS[proto]
    unneeded_exclusions = excluded_attrs - shared_attrs
    assert not unneeded_exclusions, f"Unnecessary Exclusions: {unneeded_exclusions}"
    shared_attrs -= EXCLUDED_MEMBERS[proto]

    if extra_attrs := sorted(protocol_members - shared_attrs):
        raise AssertionError(f"\nMissing attributes: {extra_attrs}")
    if missing_attrs := sorted(filter(is_admissable, shared_attrs - protocol_members)):
        raise AssertionError(
            f"\nShared members not covered by {proto}:\n\t{missing_attrs}"
        )


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
