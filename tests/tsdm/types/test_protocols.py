#!/usr/bin/env python
"""Test the Array protocol."""

import logging

import numpy
import pyarrow as pa
import torch
from numpy import ndarray
from numpy.typing import NDArray
from pandas import DataFrame, Index, Series
from torch import Tensor

from tsdm.types.protocols import Array, Table

__logger__ = logging.getLogger(__name__)


def test_table() -> None:
    """Test the Table protocol (shape and __len__ and __getitem__)."""
    LOGGER = __logger__.getChild(Table.__name__)
    LOGGER.info("Testing.")

    torch_tensor: Tensor = torch.tensor([1, 2, 3])
    torch_table: Table = torch_tensor
    assert isinstance(
        torch_table, Table
    ), f"Missing Attributes: {set(dir(Table)) - set(dir(torch_table))}"

    numpy_ndarray: NDArray = ndarray([1, 2, 3])
    numpy_table: Table = numpy_ndarray
    assert isinstance(
        numpy_table, Table
    ), f"Missing Attributes: {set(dir(Table)) - set(dir(numpy_table))}"

    pandas_frame: DataFrame = DataFrame(numpy.random.randn(3, 3))
    pandas_table: Table = pandas_frame
    assert isinstance(
        pandas_table, Table
    ), f"Missing Attributes: {set(dir(Table)) - set(dir(pandas_table))}"

    pandas_series: Series = Series([1, 2, 3])
    pandas_series_array: Table = pandas_series
    assert isinstance(
        pandas_series_array, Table
    ), f"Missing Attributes: {set(dir(Table)) - set(dir(pandas_series_array))}"

    pandas_index: Index = Index([1, 2, 3])
    pandas_index_array: Table = pandas_index
    assert isinstance(
        pandas_index_array, Table
    ), f"Missing Attributes: {set(dir(Table)) - set(dir(pandas_index_array))}"

    pyarrow_frame: pa.Table = pa.Table.from_pandas(pandas_frame)
    pyarrow_table: Table = pyarrow_frame
    assert isinstance(
        pyarrow_table, Table
    ), f"Missing Attributes: {set(dir(Table)) - set(dir(pyarrow_table))}"

    pyarrow_series: pa.Array = pa.Array.from_pandas(pandas_series)
    pyarrow_series_table: Table = pyarrow_series
    assert isinstance(
        pyarrow_table, Table
    ), f"Missing Attributes: {set(dir(Table)) - set(dir(pyarrow_series_table))}"

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


def test_array() -> None:
    """Test the Array protocol (singular dtype and ndim)."""
    torch_tensor: Tensor = torch.tensor([1, 2, 3])
    torch_array: Array = torch_tensor
    assert isinstance(
        torch_array, Array
    ), f"Missing Attributes: {set(dir(Array)) - set(dir(torch_array))}"

    numpy_ndarray: NDArray = ndarray([1, 2, 3])
    numpy_array: Array = numpy_ndarray
    assert isinstance(
        numpy_array, Array
    ), f"Missing Attributes: {set(dir(Array)) - set(dir(numpy_array))}"

    pandas_series: Series = Series([1, 2, 3])
    pandas_array2: Array = pandas_series
    assert isinstance(
        pandas_array2, Array
    ), f"Missing Attributes: {set(dir(Array)) - set(dir(pandas_array2))}"

    arrays = [torch_array, numpy_array, pandas_array2]
    shared_attrs = set.intersection(*(set(dir(arr)) for arr in arrays))
    __logger__.info("Shared attributes/methods of Arrays: %s", shared_attrs)


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_table()
    test_array()


if __name__ == "__main__":
    _main()
