"""Test the Array protocol."""

import logging
from typing import NamedTuple

import numpy
import pyarrow as pa
import torch
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from pytest import mark
from torch import Tensor
from typing_extensions import get_protocol_members

from tsdm.types.protocols import (
    Array,
    MutableArray,
    NTuple,
    NumericalArray,
    Shape,
    SupportsShape,
    assert_protocol,
)

__logger__ = logging.getLogger(__name__)


ARRAY_PROTOCOLS = (Array, NumericalArray, MutableArray)


ARRAYS = {
    "torch_tensor": torch.tensor([1, 2, 3]),
    "numpy_ndarray": numpy.ndarray([1, 2, 3]),
    "pandas_series": Series([1, 2, 3]),
    "pandas_index": Index([1, 2, 3]),
    "pandas_dataframe": DataFrame(numpy.random.randn(3, 3)),
    "pyarrow_array": pa.array([1, 2, 3]),
    "pyarrow_table": pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}),
    # "python_array": memoryview(builtin_array("i", [1, 2, 3])),
}

NUMERICAL_ARRAYS = {
    "torch_tensor": torch.tensor([1, 2, 3]),
    "numpy_ndarray": numpy.ndarray([1, 2, 3]),
    "pandas_series": Series([1, 2, 3]),
    "pandas_dataframe": DataFrame(numpy.random.randn(3, 3)),
}


def test_shape() -> None:
    """Test the Shape protocol."""
    data = [1, 2, 3]
    torch_tensor: Tensor = torch.tensor(data)
    numpy_ndarray: NDArray = numpy.ndarray(data)
    pandas_series: Series = Series(data)
    pandas_index: Index = Index(data)

    x: Shape = (1, 2, 3)
    y: Shape = torch_tensor.shape
    z: Shape = numpy_ndarray.shape
    w: Shape = pandas_series.shape
    v: Shape = pandas_index.shape
    assert isinstance(x, Shape)
    assert isinstance(y, Shape)
    assert isinstance(z, Shape)
    assert isinstance(w, Shape)
    assert isinstance(v, Shape)


def test_ntuple() -> None:
    """Test the NTuple protocol."""

    class Point(NamedTuple):
        """A point in 2D space."""

        x: float
        y: float

    # check that NamedTuples are tuples.
    assert issubclass(Point, tuple)
    assert isinstance(Point(1, 2), tuple)

    # check against plain tuple
    # assert issubclass(NTuple, tuple)
    assert not issubclass(tuple, NTuple)  # type: ignore[misc]

    assert isinstance((1, 2), tuple)
    assert not isinstance((1, 2), NTuple)

    assert issubclass(Point, NTuple)  # type: ignore[misc]
    assert issubclass(Point, tuple)
    assert not issubclass(tuple, Point)
    assert not issubclass(NTuple, Point)

    assert isinstance(Point(1, 2), NTuple)
    assert not isinstance((1, 2), Point)


def test_table() -> None:
    """Test the Table protocol (shape and __len__ and __getitem__)."""
    LOGGER = __logger__.getChild(SupportsShape.__name__)
    LOGGER.info("Testing.")

    torch_tensor: Tensor = torch.tensor([1, 2, 3])
    torch_table: SupportsShape = torch_tensor
    assert isinstance(
        torch_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(torch_table))}"

    numpy_ndarray: NDArray = numpy.ndarray([1, 2, 3])
    numpy_table: SupportsShape = numpy_ndarray
    assert isinstance(
        numpy_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(numpy_table))}"

    pandas_frame: DataFrame = DataFrame(numpy.random.randn(3, 3))
    pandas_table: SupportsShape = pandas_frame
    assert isinstance(
        pandas_table, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_table))}"

    pandas_series: Series = Series([1, 2, 3])
    pandas_series_array: SupportsShape = pandas_series
    assert isinstance(
        pandas_series_array, SupportsShape
    ), f"Missing Attributes: {set(dir(SupportsShape)) - set(dir(pandas_series_array))}"

    pandas_index: Index = Index([1, 2, 3])
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
    __logger__.info("Shared attributes/methods of Tables: %s", shared_attrs)


# @pytest.mark.parametrize("array", TEST_ARRAYS)
@mark.parametrize("protocol", ARRAY_PROTOCOLS)
def test_numerical_arrays(protocol: type) -> None:
    """Test that all arrays share the same attributes."""
    protocol_attrs = get_protocol_members(protocol)
    for key, array in NUMERICAL_ARRAYS.items():
        missing_attrs = protocol_attrs - set(dir(array))
        assert not missing_attrs, f"{key}: missing Attributes: {missing_attrs}"
        assert isinstance(array, protocol), f"{key} is not a {protocol.__name__}!"

    shared_attrs = set.intersection(
        *(set(dir(arr)) for arr in NUMERICAL_ARRAYS.values())
    )
    superfluous_attrs = shared_attrs - protocol_attrs
    print(
        f"Shared attributes/methods not covered by {protocol.__name__!r}:"
        f" {superfluous_attrs}"
    )


# NOTE: Intentionally not using parametrize to allow type-checking
def test_arrays_jointly() -> None:
    """Test the Array protocol (singular dtype and ndim)."""
    # list of all arrays
    arrays: list = []

    # test torch
    torch_tensor: Tensor = torch.tensor([1, 2, 3])
    assert_protocol(torch_tensor, Array)
    arrays.append(torch_tensor)

    # test numpy
    numpy_array: Array = numpy.ndarray([1, 2, 3])
    assert_protocol(numpy_array, Array)
    arrays.append(numpy_array)

    # test pandas Series
    pandas_series: Array = Series([1, 2, 3])
    assert_protocol(pandas_series, Array)
    arrays.append(pandas_series)

    # test pandas Index
    pandas_index: Array = Index([1, 2, 3])
    assert_protocol(pandas_index, Array)
    arrays.append(pandas_index)

    # test dataframe
    pandas_frame: Array = DataFrame(numpy.random.randn(3, 3))
    assert_protocol(pandas_frame, Array)
    arrays.append(pandas_frame)

    # test pyarrow.Array
    pyarrow_array: Array = pa.array([1, 2, 3])
    assert_protocol(pyarrow_array, Array)
    arrays.append(pyarrow_array)

    # test pyarrow.Table
    pyarrow_table: Array = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_protocol(pyarrow_table, Array)
    arrays.append(pyarrow_table)

    # # test python array (missing: __array__)
    # python_array: Array = memoryview(builtin_array("i", [1, 2, 3]))
    # # assert_protocol(python_array, Array)
    # arrays.append(python_array)

    # test combined
    shared_attrs = set.intersection(*(set(dir(arr)) for arr in arrays))
    superfluous_attrs = shared_attrs - set(dir(Array))
    assert not superfluous_attrs, f"Shared attributes/methods: {superfluous_attrs}"
    __logger__.info("Shared attributes/methods of Arrays: %s", shared_attrs)


@mark.parametrize("name", ARRAYS)
def test_array_all(name: str) -> None:
    """Test the Array protocol (singular dtype and ndim)."""
    assert_protocol(ARRAYS[name], Array)
