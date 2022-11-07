#!/usr/bin/env python
"""Test the Array protocol."""

import logging

import torch
from numpy import ndarray
from numpy.typing import NDArray
from pandas import Series
from torch import Tensor

from tsdm.utils.types.protocols import Array


def test_array():
    """Test the Array protocol."""
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

    # Misses .dtype, has .dtypes instead
    # pandas_frame: DataFrame = DataFrame([1, 2, 3])
    # pandas_array: Array = pandas_frame
    # assert isinstance(
    #     pandas_array, Array
    # ), f"Missing Attributes: {set(dir(Array)) - set(dir(pandas_array))}"


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    test_array()


if __name__ == "__main__":
    _main()
