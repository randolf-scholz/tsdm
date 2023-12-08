"""Test the Array protocol."""

import logging
from typing import Any

import numpy
import pandas
import polars
import pyarrow
import torch
from numpy.typing import NDArray
from pytest import mark
from typing_extensions import get_protocol_members

from tsdm.types.protocols import (
    ArrayKind,
    MutableArray,
    NumericalArray,
    SeriesKind,
    SupportsArray,
    SupportsShape,
    TableKind,
    assert_protocol,
)

__logger__ = logging.getLogger(__name__)


ARRAY_PROTOCOLS = (ArrayKind, NumericalArray, MutableArray)

_SERIES_DATA = ["a", "b", "c"]
_ARRAY_DATA = [[1, 2, 3], [4, 5, 6]]
_TABLE_DATA = {
    "integers": [1, 2, 3, 4],
    "floats": [1.1, 2.2, 3.3, 4.4],
    "strings": ["a", "b", "c", "d'"],
}

NP_ARRAY = numpy.array(_ARRAY_DATA)
PD_INDEX = pandas.Index(_SERIES_DATA)
PD_SERIES = pandas.Series(_SERIES_DATA)
PD_DATAFRAME = pandas.DataFrame(_TABLE_DATA)
PL_SERIES = polars.Series(_SERIES_DATA)
PL_DATAFRAME = polars.DataFrame(_TABLE_DATA)
PA_ARRAY = pyarrow.array(_SERIES_DATA)
PA_TABLE = pyarrow.table(_TABLE_DATA)
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
    "torch_tensor": TORCH_TENSOR,
    # "python_array": memoryview(builtin_array("i", [1, 2, 3])),
}


SERIES: dict[str, SeriesKind[str]] = {
    # "python_array": memoryview(builtin_array("i", [1, 2, 3])),
    "pandas_index": PD_INDEX,
    "pandas_series": PD_SERIES,
    "polars_series": PL_SERIES,
    "pyarrow_array": PA_ARRAY,
}

TABLES: dict[str, TableKind] = {
    "pandas_dataframe": PD_DATAFRAME,
    "polars_dataframe": PL_DATAFRAME,
    "pyarrow_table": PA_TABLE,
}

ARRAYS: dict[str, ArrayKind] = {
    "numpy_ndarray": NP_ARRAY,
    "pandas_dataframe": PD_DATAFRAME,
    "pandas_index": PD_INDEX,
    "pandas_series": PD_SERIES,
    # "polars_dataframe": PL_DATAFRAME,
    "polars_series": PL_SERIES,
    "torch_tensor": TORCH_TENSOR,
    # "pyarrow_array": PA_ARRAY,
    "pyarrow_table": PA_TABLE,
    # "python_array": memoryview(builtin_array("i", [1, 2, 3])),
}

NUMERICAL_ARRAYS: dict[str, NumericalArray] = {
    "numpy_ndarray": NP_ARRAY,
    "pandas_dataframe": PD_DATAFRAME,
    "pandas_series": PD_SERIES,
    # "polars_series": PL_SERIES,  # missing: ndim
    # "polars_dataframe": PL_DATAFRAME, # missing: lots
    "torch_tensor": TORCH_TENSOR,
}

MUTABLE_ARRAYS: dict[str, MutableArray] = {
    "numpy_ndarray": NP_ARRAY,
    "pandas_dataframe": PD_DATAFRAME,
    "pandas_series": PD_SERIES,
    "torch_tensor": TORCH_TENSOR,
}

EXAMPLES: dict[type, dict[str, Any]] = {
    SeriesKind: SERIES,
    TableKind: TABLES,
    ArrayKind: ARRAYS,
    NumericalArray: NUMERICAL_ARRAYS,
    MutableArray: MUTABLE_ARRAYS,
}


@mark.parametrize("name", SUPPORTS_ARRAYS)
def test_supports_array(name: str) -> None:
    """Test the SupportsArray protocol."""
    obj = SUPPORTS_ARRAYS[name]
    assert isinstance(obj, SupportsArray)
    assert issubclass(obj.__class__, SupportsArray)
    assert isinstance(obj.__array__(), numpy.ndarray)


@mark.parametrize("name", SERIES)
def test_series(name: str) -> None:
    """Test the Series protocol."""
    series = SERIES[name]
    cls = series.__class__
    assert isinstance(series, SeriesKind)
    assert not isinstance(series, TableKind)

    # check methods
    attrs = set(get_protocol_members(SeriesKind))

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
    """Test the Table protocol."""
    table = TABLES[name]
    assert isinstance(table, TableKind)
    assert not isinstance(table, SeriesKind)

    # check methods
    attrs = set(get_protocol_members(TableKind))

    assert isinstance(table.__array__(), numpy.ndarray)
    attrs.remove("__array__")

    table.__dataframe__()
    attrs.remove("__dataframe__")

    assert isinstance(len(table), int)
    attrs.remove("__len__")

    assert isinstance(table.shape, tuple)
    assert len(table.shape) == 2
    assert isinstance(table.shape[0], int) and isinstance(table.shape[1], int)
    attrs.remove("shape")

    assert isinstance(table["floats"], SeriesKind)
    attrs.remove("__getitem__")

    # check that all attributes are tested
    assert not attrs, f"Forgot to test: {attrs}!"


@mark.parametrize("name", ARRAYS)
def test_array(name: str) -> None:
    """Test the Array protocol."""
    assert_protocol(ARRAYS[name], ArrayKind)


@mark.parametrize("name", NUMERICAL_ARRAYS)
def test_numerical_array(name: str) -> None:
    """Test the NumericalArray protocol."""
    assert_protocol(NUMERICAL_ARRAYS[name], NumericalArray)


@mark.parametrize("name", MUTABLE_ARRAYS)
def test_mutable_array(name: str) -> None:
    """Test the MutableArray protocol."""
    assert_protocol(MUTABLE_ARRAYS[name], MutableArray)


def test_shared_attrs() -> None:
    """Test which shared attributes exist that are not covered by protocols."""
    print("\nShared Attributes not covered by protocols:")
    for proto, examples in EXAMPLES.items():
        shared_attrs = set.intersection(*(set(dir(s)) for s in examples.values()))
        superfluous_attrs = sorted(shared_attrs - set(dir(proto)))
        print(f"\n\t{proto.__name__!r}:\n\t{superfluous_attrs}")


def test_series_joint_attrs() -> None:
    shared_attrs = set.intersection(*(set(dir(s)) for s in SERIES.values()))
    superfluous_attrs = shared_attrs - set(dir(SeriesKind))
    print(f"\nShared members not covered by SeriesKind:\n\t{superfluous_attrs}")


def test_table_joint_attrs() -> None:
    shared_attrs = set.intersection(*(set(dir(t)) for t in TABLES.values()))
    superfluous_attrs = shared_attrs - set(dir(TableKind))
    print(f"\nShared members not covered by TableKind:\n\t{superfluous_attrs}")


def test_array_joint_attrs() -> None:
    shared_attrs = set.intersection(*(set(dir(a)) for a in ARRAYS.values()))
    superfluous_attrs = shared_attrs - set(dir(ArrayKind))
    print(f"\nShared members not covered by ArrayKind:\n\t{superfluous_attrs}")


def test_table_manual() -> None:
    """Test the Table protocol (shape and __len__ and __getitem__)."""
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

    pandas_frame: pandas.DataFrame = pandas.DataFrame(numpy.random.randn(3, 3))
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


# # @pytest.mark.parametrize("array", TEST_ARRAYS)
# @mark.parametrize("protocol", ARRAY_PROTOCOLS)
# def test_numerical_arrays(protocol: type) -> None:
#     """Test that all arrays share the same attributes."""
#     protocol_attrs = get_protocol_members(protocol)
#     for key, array in NUMERICAL_ARRAYS.items():
#         missing_attrs = protocol_attrs - set(dir(array))
#         assert not missing_attrs, f"{key}: missing Attributes: {missing_attrs}"
#         assert isinstance(array, protocol), f"{key} is not a {protocol.__name__}!"
#
#     shared_attrs = set.intersection(
#         *(set(dir(arr)) for arr in NUMERICAL_ARRAYS.values())
#     )
#     superfluous_attrs = shared_attrs - protocol_attrs
#     print(
#         f"\nShared members not covered by {protocol.__name__!r}:\n\t{superfluous_attrs}"
#     )


# NOTE: Intentionally not using parametrize to allow type-checking
def test_arrays_jointly() -> None:
    """Test the Array protocol (singular dtype and ndim)."""
    # list of all arrays
    arrays: list = []

    # test torch
    torch_tensor: torch.Tensor = torch.tensor([1, 2, 3])
    assert_protocol(torch_tensor, ArrayKind)
    arrays.append(torch_tensor)

    # test numpy
    numpy_array: ArrayKind = numpy.array([1, 2, 3])
    assert_protocol(numpy_array, ArrayKind)
    arrays.append(numpy_array)

    # test pandas Series
    pandas_series: ArrayKind = pandas.Series([1, 2, 3])
    assert_protocol(pandas_series, ArrayKind)
    arrays.append(pandas_series)

    # test pandas Index
    pandas_index: ArrayKind = pandas.Index([1, 2, 3])
    assert_protocol(pandas_index, ArrayKind)
    arrays.append(pandas_index)

    # test dataframe
    pandas_frame: ArrayKind = pandas.DataFrame(numpy.random.randn(3, 3))
    assert_protocol(pandas_frame, ArrayKind)
    arrays.append(pandas_frame)

    # test pyarrow.Array  # FIXME: missing .shape
    # pyarrow_array: ArrayKind = pyarrow.array([1, 2, 3])
    # assert_protocol(pyarrow_array, ArrayKind)
    # arrays.append(pyarrow_array)

    # test pyarrow.Table
    pyarrow_table: ArrayKind = pyarrow.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_protocol(pyarrow_table, ArrayKind)
    arrays.append(pyarrow_table)

    # # test python array (missing: __array__)
    # python_array: Array = memoryview(builtin_array("i", [1, 2, 3]))
    # # assert_protocol(python_array, Array)
    # arrays.append(python_array)

    # test combined
    shared_attrs = set.intersection(*(set(dir(arr)) for arr in arrays))
    superfluous_attrs = shared_attrs - set(dir(ArrayKind))
    assert not superfluous_attrs, f"\nShared members:\n\t{superfluous_attrs}"
    __logger__.info("\nShared members of Arrays: %s", shared_attrs)
