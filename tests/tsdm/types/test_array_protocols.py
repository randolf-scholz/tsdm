"""Test the Array protocol."""

import logging
from array import array as python_array
from collections.abc import Collection
from typing import Any

import numpy
import pandas
import polars
import pyarrow
import torch
from numpy.typing import NDArray
from pytest import mark
from typing_extensions import get_protocol_members

from tsdm.testing import assert_protocol
from tsdm.types.protocols import (
    ArrayKind,
    MutableArray,
    NumericalArray,
    SeriesKind,
    SupportsArray,
    SupportsShape,
    TableKind,
)

__logger__ = logging.getLogger(__name__)
RNG = numpy.random.default_rng()
ARRAY_PROTOCOLS = (ArrayKind, NumericalArray, MutableArray)

_SERIES_DATA = ["a", "b", "c"]
_ARRAY_DATA = [[1, 2, 3], [4, 5, 6]]
_TABLE_DATA = {
    "integers": [1, 2, 3, 4],
    "floats": [1.1, 2.2, 3.3, 4.4],
    "strings": ["a", "b", "c", "d'"],
}

NP_ARRAY = numpy.array(_ARRAY_DATA)
PA_ARRAY = pyarrow.array(_SERIES_DATA)
PA_TABLE = pyarrow.table(_TABLE_DATA)
PD_DATAFRAME = pandas.DataFrame(_TABLE_DATA)
PD_INDEX = pandas.Index(_SERIES_DATA)
PD_SERIES = pandas.Series(_SERIES_DATA)
PL_DATAFRAME = polars.DataFrame(_TABLE_DATA)
PL_SERIES = polars.Series(_SERIES_DATA)
PY_ARRAY = memoryview(python_array("i", [1, 2, 3]))
TORCH_TENSOR = torch.tensor(_ARRAY_DATA)

SUPPORTS_ARRAYS: dict[str, SupportsArray] = {
    "numpy_ndarray": NP_ARRAY,
    "pandas_dataframe": PD_DATAFRAME,
    "pandas_index": PD_INDEX,
    "pandas_series": PD_SERIES,
    "polars_dataframe": PL_DATAFRAME,
    "polars_series": PL_SERIES,
    "pyarrow_array": PA_ARRAY,
    "pyarrow_table": PA_TABLE,
    # "python_array": PY_ARRAY,
    "torch_tensor": TORCH_TENSOR,
}

SERIES: dict[str, SeriesKind[str]] = {
    # "numpy_ndarray": NP_ARRAY,
    # "pandas_dataframe": PD_DATAFRAME,
    "pandas_index": PD_INDEX,
    "pandas_series": PD_SERIES,
    # "polars_dataframe": PL_DATAFRAME,
    "polars_series": PL_SERIES,
    "pyarrow_array": PA_ARRAY,
    # "pyarrow_table": PA_TABLE,
    # "python_array": PY_ARRAY,
    # "torch_tensor": TORCH_TENSOR,
}

TABLES: dict[str, TableKind] = {
    # "numpy_ndarray": NP_ARRAY,    # missing __dataframe__
    "pandas_dataframe": PD_DATAFRAME,
    # "pandas_index": PD_INDEX,    # missing __dataframe__
    # "pandas_series": PD_SERIES,    # missing __dataframe__
    "polars_dataframe": PL_DATAFRAME,
    # "polars_series": PL_SERIES,    # missing __dataframe__
    # "pyarrow_array": PA_ARRAY,    # missing __dataframe__
    "pyarrow_table": PA_TABLE,
    # "python_array": PY_ARRAY,  # missing __dataframe__
    # "torch_tensor": TORCH_TENSOR,    # missing __dataframe__
}

ARRAYS: dict[str, ArrayKind] = {
    "numpy_ndarray": NP_ARRAY,
    "pandas_dataframe": PD_DATAFRAME,
    "pandas_index": PD_INDEX,
    "pandas_series": PD_SERIES,
    "polars_dataframe": PL_DATAFRAME,
    "polars_series": PL_SERIES,
    # "pyarrow_array": PA_ARRAY,  # missing shape
    "pyarrow_table": PA_TABLE,
    # "python_array": PY_ARRAY,  # missing __array__
    "torch_tensor": TORCH_TENSOR,
}

NUMERICAL_ARRAYS: dict[str, NumericalArray] = {
    "numpy_ndarray": NP_ARRAY,
    "pandas_dataframe": PD_DATAFRAME,
    "pandas_index": PD_INDEX,
    "pandas_series": PD_SERIES,
    # "polars_dataframe": PL_DATAFRAME,  # missing r-methods
    # "polars_series": PL_SERIES,  # missing ndim
    # "pyarrow_array": PA_ARRAY,
    # "pyarrow_table": PA_TABLE,
    # "python_array": PY_ARRAY,
    "torch_tensor": TORCH_TENSOR,
}

MUTABLE_ARRAYS: dict[str, MutableArray] = {
    "numpy_ndarray": NP_ARRAY,
    "pandas_dataframe": PD_DATAFRAME,
    # "pandas_index": PD_INDEX,  # missing i-methods
    "pandas_series": PD_SERIES,
    # "polars_dataframe": PL_DATAFRAME,  # missing i-methods
    # "polars_series": PL_SERIES,  # missing i-methods
    # "pyarrow_array": PA_ARRAY,  # missing i-methods
    # "pyarrow_table": PA_TABLE,  # missing i-methods
    # "python_array": PY_ARRAY,
    "torch_tensor": TORCH_TENSOR,
}

EXAMPLES: dict[type, dict[str, Any]] = {
    SeriesKind: SERIES,
    TableKind: TABLES,
    ArrayKind: ARRAYS,
    NumericalArray: NUMERICAL_ARRAYS,
    MutableArray: MUTABLE_ARRAYS,
}

# frozensets of attributes
TABLE_ATTRS = get_protocol_members(TableKind)
SERIES_ATTRS = get_protocol_members(SeriesKind)
ARRAY_ATTRS = get_protocol_members(ArrayKind)
NUMERICAL_ARRAY_ATTRS = get_protocol_members(NumericalArray)
MUTABLE_ARRAY_ATTRS = get_protocol_members(MutableArray)


@mark.parametrize("name", SUPPORTS_ARRAYS)
def test_supports_array(name: str) -> None:
    r"""Test the SupportsArray protocol."""
    obj = SUPPORTS_ARRAYS[name]
    assert isinstance(obj, SupportsArray)
    assert issubclass(obj.__class__, SupportsArray)
    assert isinstance(obj.__array__(), numpy.ndarray)


@mark.parametrize("name", SERIES)
def test_series(name: str) -> None:
    r"""Test the Series protocol."""
    series = SERIES[name]
    cls = series.__class__
    assert isinstance(series, SeriesKind), SERIES_ATTRS - set(dir(series))
    assert not isinstance(series, TableKind)

    # check methods
    attrs = set(SERIES_ATTRS)

    assert isinstance(series.__array__(), numpy.ndarray)
    attrs.remove("__array__")

    assert isinstance(len(series), int)
    attrs.remove("__len__")

    for x in series:
        assert isinstance(x, str | pyarrow.StringScalar)
    attrs.remove("__iter__")

    assert isinstance(series[0], str | pyarrow.StringScalar)
    assert isinstance(series[0:2], cls)
    attrs.remove("__getitem__")

    assert isinstance(series.unique(), cls | numpy.ndarray)
    attrs.remove("unique")

    series.value_counts()
    attrs.remove("value_counts")

    assert isinstance(series.take([0, 0, 2]), cls)
    attrs.remove("take")

    # check that all attributes are tested
    assert not attrs, f"Forgot to test: {attrs}!"


@mark.parametrize("name", TABLES)
def test_table(name: str) -> None:
    r"""Test the Table protocol."""
    table = TABLES[name]
    assert isinstance(table, TableKind), TABLE_ATTRS - set(dir(table))
    assert not isinstance(table, SeriesKind)

    # check methods
    attrs = set(TABLE_ATTRS)

    assert isinstance(table.__array__(), numpy.ndarray)
    attrs.remove("__array__")

    table.__dataframe__()
    attrs.remove("__dataframe__")

    assert isinstance(len(table), int)
    attrs.remove("__len__")

    assert isinstance(table.columns, Collection)
    assert all(isinstance(col, str) for col in table.columns) or all(
        isinstance(col, SeriesKind) for col in table.columns
    )
    attrs.remove("columns")

    assert isinstance(table.shape, tuple)
    assert len(table.shape) == 2
    assert isinstance(table.shape[0], int)
    assert isinstance(table.shape[1], int)
    attrs.remove("shape")

    assert isinstance(table["floats"], SeriesKind)
    attrs.remove("__getitem__")

    # check that all attributes are tested
    assert not attrs, f"Forgot to test: {attrs}!"


@mark.parametrize("name", ARRAYS)
def test_array(name: str) -> None:
    r"""Test the Array protocol."""
    array = ARRAYS[name]
    assert_protocol(array, ArrayKind)


@mark.parametrize("name", NUMERICAL_ARRAYS)
def test_numerical_array(name: str) -> None:
    r"""Test the NumericalArray protocol."""
    numerical_array = NUMERICAL_ARRAYS[name]
    assert_protocol(numerical_array, NumericalArray)


@mark.parametrize("name", MUTABLE_ARRAYS)
def test_mutable_array(name: str) -> None:
    r"""Test the MutableArray protocol."""
    mutable_array = MUTABLE_ARRAYS[name]
    assert_protocol(mutable_array, MutableArray)


def test_shared_attrs() -> None:
    r"""Test which shared attributes exist that are not covered by protocols."""
    print("\nShared Attributes not covered by protocols:")
    for proto, examples in EXAMPLES.items():
        protocol_members = get_protocol_members(proto)
        shared_attrs = set.intersection(*(set(dir(s)) for s in examples.values()))
        superfluous_attrs = sorted(shared_attrs - protocol_members)
        print(f"\n\t{proto.__name__!r}:\n\t{superfluous_attrs}")
        missing_attrs = sorted(protocol_members - shared_attrs)
        assert not missing_attrs, f"{proto}: not all examples have: {missing_attrs}!"


def test_joint_attrs_series() -> None:
    shared_attrs = set.intersection(*(set(dir(s)) for s in SERIES.values()))
    series_members = get_protocol_members(SeriesKind)
    superfluous_attrs = sorted(shared_attrs - series_members)
    print(f"\nShared members not covered by SeriesKind:\n\t{superfluous_attrs}")
    missing_attrs = sorted(series_members - shared_attrs)
    assert not missing_attrs, f"Missing attributes: {missing_attrs}"


def test_joint_attrs_table() -> None:
    shared_attrs = set.intersection(*(set(dir(t)) for t in TABLES.values()))
    table_members = get_protocol_members(TableKind)
    superfluous_attrs = sorted(shared_attrs - table_members)
    print(f"\nShared members not covered by TableKind:\n\t{superfluous_attrs}")
    missing_attrs = sorted(table_members - shared_attrs)
    assert not missing_attrs, f"Missing attributes: {missing_attrs}"


def test_joint_attrs_array() -> None:
    shared_attrs = set.intersection(*(set(dir(a)) for a in ARRAYS.values()))
    array_members = get_protocol_members(ArrayKind)
    superfluous_attrs = sorted(shared_attrs - array_members)
    print(f"\nShared members not covered by ArrayKind:\n\t{superfluous_attrs}")
    missing_attrs = sorted(array_members - shared_attrs)
    assert not missing_attrs, f"Missing attributes: {missing_attrs}"


def test_joint_attrs_numerical_array() -> None:
    shared_attrs = set.intersection(*(set(dir(a)) for a in NUMERICAL_ARRAYS.values()))
    numerical_array_members = get_protocol_members(NumericalArray)
    superfluous_attrs = sorted(shared_attrs - numerical_array_members)
    print(f"\nShared members not covered by NumericalArray:\n\t{superfluous_attrs}")
    missing_attrs = sorted(numerical_array_members - shared_attrs)
    assert not missing_attrs, f"Missing attributes: {missing_attrs}"


def test_joint_attrs_mutable_array() -> None:
    shared_attrs = set.intersection(*(set(dir(a)) for a in MUTABLE_ARRAYS.values()))
    mutable_array_members = get_protocol_members(MutableArray)
    superfluous_attrs = sorted(shared_attrs - mutable_array_members)
    print(f"\nShared members not covered by MutableArray:\n\t{superfluous_attrs}")
    missing_attrs = sorted(mutable_array_members - shared_attrs)
    assert not missing_attrs, f"Missing attributes: {missing_attrs}"


def test_table_manual() -> None:
    r"""Test the Table protocol (shape and __len__ and __getitem__)."""
    LOGGER = __logger__.getChild(SupportsShape.__name__)
    LOGGER.info("Testing.")

    torch_tensor: torch.Tensor = torch.tensor([1, 2, 3])
    torch_table: SupportsShape = torch_tensor
    assert isinstance(
        torch_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(torch_table))}"

    numpy_ndarray: NDArray = numpy.array([1, 2, 3])
    numpy_table: SupportsShape = numpy_ndarray
    assert isinstance(
        numpy_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(numpy_table))}"

    pandas_frame: pandas.DataFrame = pandas.DataFrame(RNG.normal(size=(3, 3)))
    pandas_table: SupportsShape = pandas_frame
    assert isinstance(
        pandas_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_table))}"

    pandas_series: pandas.Series = pandas.Series([1, 2, 3])
    pandas_series_array: SupportsShape = pandas_series
    assert isinstance(
        pandas_series_array, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_series_array))}"

    pandas_index: pandas.Index = pandas.Index([1, 2, 3])
    pandas_index_array: SupportsShape = pandas_index
    assert isinstance(
        pandas_index_array, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_index_array))}"

    pyarrow_frame: pyarrow.Table = pyarrow.Table.from_pandas(pandas_frame)
    pyarrow_table: SupportsShape = pyarrow_frame
    assert isinstance(
        pyarrow_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pyarrow_table))}"

    pyarrow_series: pyarrow.Array = pyarrow.Array.from_pandas(pandas_series)
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
