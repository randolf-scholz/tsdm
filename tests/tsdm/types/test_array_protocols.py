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
    NumericalTensor,
    SeriesKind,
    SupportsArray,
    SupportsShape,
    TableKind,
)

__logger__ = logging.getLogger(__name__)
RNG = np.random.default_rng()
ARRAY_PROTOCOLS = (ArrayKind, NumericalArray, MutableArray)

_SERIES_DATA = ["a", "b", "c"]
_ARRAY_DATA = [[1, 2, 3], [4, 5, 6]]
_TABLE_DATA = {
    "integers": [1, 2, 3, 4],
    "floats": [1.1, 2.2, 3.3, 4.4],
    "strings": ["a", "b", "c", "d'"],
}

NP_ARRAY = np.array(_ARRAY_DATA)
PA_ARRAY = pa.array(_SERIES_DATA)
PA_TABLE = pa.table(_TABLE_DATA)
PD_DATAFRAME = pd.DataFrame(_TABLE_DATA)
PD_INDEX = pd.Index(_SERIES_DATA)
PD_SERIES = pd.Series(_SERIES_DATA)
PL_DATAFRAME = pl.DataFrame(_TABLE_DATA)
PL_SERIES = pl.Series(_SERIES_DATA)
PY_ARRAY = memoryview(python_array("i", [1, 2, 3]))
TORCH_TENSOR = torch.tensor(_ARRAY_DATA)

TEST_OBJECTS = {
    "python_array"     : PY_ARRAY,
    "numpy_ndarray"    : NP_ARRAY,
    "pandas_index"     : PD_INDEX,
    "pandas_series"    : PD_SERIES,
    "pandas_dataframe" : PD_DATAFRAME,
    "polars_series"    : PL_SERIES,
    "polars_dataframe" : PL_DATAFRAME,
    "pyarrow_array"    : PA_ARRAY,
    "pyarrow_table"    : PA_TABLE,
    "torch_tensor"     : TORCH_TENSOR,
}  # fmt: skip

SUPPORTS_ARRAYS: dict[str, SupportsArray] = {
    "numpy_ndarray"    : NP_ARRAY,
    "pandas_dataframe" : PD_DATAFRAME,
    "pandas_index"     : PD_INDEX,
    "pandas_series"    : PD_SERIES,
    "polars_dataframe" : PL_DATAFRAME,
    "polars_series"    : PL_SERIES,
    "pyarrow_array"    : PA_ARRAY,
    "pyarrow_table"    : PA_TABLE,
    "torch_tensor"     : TORCH_TENSOR,
}  # fmt: skip

SERIES: dict[str, SeriesKind[str]] = {
    "pandas_index"  : PD_INDEX,
    "pandas_series" : PD_SERIES,
    "polars_series" : PL_SERIES,
    "pyarrow_array" : PA_ARRAY,
}  # fmt: skip

TABLES: dict[str, TableKind] = {
    "pandas_dataframe" : PD_DATAFRAME,
    "polars_dataframe" : PL_DATAFRAME,
    "pyarrow_table"    : PA_TABLE,
}  # fmt: skip

ARRAYS: dict[str, ArrayKind] = {
    "numpy_ndarray"    : NP_ARRAY,
    "pandas_dataframe" : PD_DATAFRAME,
    "pandas_index"     : PD_INDEX,
    "pandas_series"    : PD_SERIES,
    "polars_dataframe" : PL_DATAFRAME,
    "polars_series"    : PL_SERIES,
    "pyarrow_table"    : PA_TABLE,
    "torch_tensor"     : TORCH_TENSOR,
}  # fmt: skip

NUMERICAL_ARRAYS: dict[str, NumericalArray] = {
    "numpy_ndarray"    : NP_ARRAY,
    "pandas_dataframe" : PD_DATAFRAME,
    "pandas_index"     : PD_INDEX,
    "pandas_series"    : PD_SERIES,
    "polars_series"    : PL_SERIES,
    "torch_tensor"     : TORCH_TENSOR,
}  # fmt: skip

x: NumericalArray = PL_SERIES


NUMERICAL_TENSORS: dict[str, NumericalTensor] = {
    "numpy_ndarray"    : NP_ARRAY,
    "pandas_index"     : PD_INDEX,
    "pandas_series"    : PD_SERIES,
    "polars_series"    : PL_SERIES,
    "torch_tensor"     : TORCH_TENSOR,
}  # fmt: skip

MUTABLE_ARRAYS: dict[str, MutableArray] = {
    "numpy_ndarray"    : NP_ARRAY,
    "pandas_dataframe" : PD_DATAFRAME,
    "pandas_series"    : PD_SERIES,
    "torch_tensor"     : TORCH_TENSOR,
}  # fmt: skip

EXAMPLES: dict[type, dict[str, Any]] = {
    SeriesKind     : SERIES,
    TableKind      : TABLES,
    ArrayKind      : ARRAYS,
    NumericalArray : NUMERICAL_ARRAYS,
    NumericalTensor: NUMERICAL_TENSORS,
    MutableArray   : MUTABLE_ARRAYS,
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
    ArrayKind      : set(),
    SeriesKind     : {"diff", "to_numpy", "value_counts", "view"},
    TableKind      : {"columns", "join", "drop", "filter"},
    NumericalArray : set(),
    NumericalTensor: {"view"},
    MutableArray   : {
        "T",
        "clip", "cumprod", "cumsum", "dot",
        "mean", "ndim", "prod",
        "size", "squeeze", "std", "sum",
        "swapaxes", "transpose", "var",
    },
}  # fmt: skip
r"""Excluded members for each protocol."""


def is_admissable(name: str) -> bool:
    r"""Check if the name is admissable."""
    return not name.startswith("_") or name in DUNDER_ARITHMETIC


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
    assert isinstance(series, SeriesKind)

    attrs = set(get_protocol_members(SeriesKind)) - DUNDER_ARITHMETIC

    assert isinstance(series.__array__(), np.ndarray)
    attrs.remove("__array__")

    assert isinstance(len(series), int)
    attrs.remove("__len__")

    for x in series:
        assert isinstance(x, str | pa.StringScalar)
    attrs.remove("__iter__")

    assert isinstance(series[0], str | pa.StringScalar)
    assert isinstance(series[0:2], cls)
    attrs.remove("__getitem__")

    assert isinstance(series.unique(), cls | np.ndarray)
    attrs.remove("unique")

    assert series.equals(series)
    attrs.remove("equals")

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

    assert isinstance(table["floats"], SeriesKind)
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
