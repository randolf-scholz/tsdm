#!/usr/bin/env python
"""Test the Array protocol."""

import logging
from typing import NamedTuple

import numpy
import pyarrow as pa
import pytest
import torch
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from torch import Tensor
from typing_extensions import get_protocol_members

from tsdm.types.protocols import (
    Array,
    MutableArray,
    NTuple,
    NumericalArray,
    Shape,
    SupportsShape,
)

__logger__ = logging.getLogger(__name__)


def test_shape() -> None:
    data = [1, 2, 3]
    torch_tensor: Tensor = torch.tensor(data)
    numpy_ndarray: NDArray = ndarray(data)
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

    numpy_ndarray: NDArray = ndarray([1, 2, 3])
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
@pytest.mark.parametrize("protocol", (Array, NumericalArray, MutableArray))
def test_shared_attrs(protocol: type) -> None:
    protocol_attrs = get_protocol_members(protocol)
    data = [1, 2, 3]
    arrays = {
        "torch_tensor": torch.tensor(data),
        "numpy_ndarray": ndarray(data),
        "pandas_series": Series(data),
        # "pyarrow_array": pa.array(data),
    }
    for key, array in arrays.items():
        missing_attrs = protocol_attrs - set(dir(array))
        assert not missing_attrs, f"{key}: missing Attributes: {missing_attrs}"
        assert isinstance(array, protocol), f"{key} is not a {protocol.__name__}!"

    shared_attrs = set.intersection(*(set(dir(arr)) for arr in arrays.values()))
    superfluous_attrs = shared_attrs - protocol_attrs
    print(
        f"Shared attributes/methods not covered by {protocol.__name__!r}:"
        f" {superfluous_attrs}"
    )


def test_array() -> None:
    """Test the Array protocol (singular dtype and ndim)."""
    # test torch
    torch_tensor: Tensor = torch.tensor([1, 2, 3])
    torch_array: Array = torch_tensor
    assert isinstance(
        torch_array, Array
    ), f"Missing Attributes: {set(dir(Array)) - set(dir(torch_array))}"

    # test numpy
    numpy_ndarray: NDArray[numpy.int_] = ndarray([1, 2, 3])
    numpy_array: Array[numpy.int_] = numpy_ndarray
    assert isinstance(
        numpy_array, Array
    ), f"Missing Attributes: {set(dir(Array)) - set(dir(numpy_array))}"

    # test pandas Series
    pandas_series: Series = Series([1, 2, 3])
    pandas_array: Array = pandas_series
    assert isinstance(
        pandas_array, Array
    ), f"Missing Attributes: {set(dir(Array)) - set(dir(pandas_array))}"

    # test pandas Index
    pandas_index: Index = Index([1, 2, 3])
    pandas_array2: Array = pandas_index

    # test combined
    arrays = [torch_array, numpy_array, pandas_array, pandas_array2]
    shared_attrs = set.intersection(*(set(dir(arr)) for arr in arrays))
    __logger__.info("Shared attributes/methods of Arrays: %s", shared_attrs)


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_shared_attrs()
    test_table()
    test_array()


if __name__ == "__main__":
    _main()
