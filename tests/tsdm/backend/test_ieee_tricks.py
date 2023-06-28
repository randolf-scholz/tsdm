#!/usr/bin/env python
"""This module contains tests for some tricks with tensor-types.

Often, we need some constant tensor that is the same shape/device as given tensor.
Typical values are zeros, ones, NaNs, ±inf, True, False, etc.

Sometimes we would like to keep NaNs.

There are some tricks to create such tensors in a backend-agnostic way, using IEEE arithmetic.

`is_nan` can be tested via `x != x`.
"""

from typing import Any

import numpy
import pandas
import torch
from pytest import mark

from tsdm.backend.universal import false_like, true_like
from tsdm.types.protocols import SelfMap


@mark.parametrize("formula", [lambda z: z**0])
def test_make_ones_like(formula: SelfMap) -> None:
    """Analogous to `ones_like`.

    Candidates for creating ones are:
    - `x**0`  gets rid of NANs
    - `x / x` keeps NANs, introduces NaNs for inf

    However, `x**0` is not implemented for time-delta types.
    What could work
    """
    data = numpy.array([float("-inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")])
    x: Any
    result: Any
    expected: Any

    # torch
    x = torch.tensor(data)
    result = formula(x)
    expected = torch.ones_like(x)
    assert all(result == expected)
    del x, result, expected

    # pandas
    x = pandas.Series(data)
    result = formula(x)
    expected = pandas.Series(numpy.ones_like(x))
    assert all(result == expected)

    # numpy
    x = numpy.array(data)
    result = formula(x)
    expected = numpy.ones_like(x)
    assert all(result == expected)

    # # test with time-delta type
    # data = numpy.array(data) * numpy.timedelta64(1, "s")
    #
    # # pandas
    # x = pandas.Series(data)
    # result = formula(x)
    # expected = pandas.Series(numpy.ones_like(x))
    # assert all(result == expected)
    #
    # # numpy
    # x = numpy.array(data)
    # result = formula(x)
    # expected = numpy.ones_like(x)
    # assert all(result == expected)


@mark.parametrize("formula", [lambda z: z**0 - z**0])
def test_zeros_like(formula: SelfMap) -> None:
    """Analogous to `zeros_like`.

    For creating zeros there are multiple good candidates:
    - `x - x`  # keeps NANs
    - `x**0 - x**0`  # gets rid of NANs
    - `0 * x`      # keeps NANs, introduces NaNs for inf
    - `0 ** x`  # doesn't for negatives, keeps NANs
    """
    data = numpy.array([float("-inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")])
    x: Any
    result: Any
    expected: Any

    # torch
    x = torch.tensor(data)
    result = formula(x)
    expected = torch.zeros_like(x)
    assert all(result == expected)

    # pandas
    x = pandas.Series(data)
    result = formula(x)
    expected = pandas.Series(numpy.zeros_like(x))
    assert all(result == expected)

    # numpy
    x = numpy.array(data)
    result = formula(x)
    expected = numpy.zeros_like(x)
    assert all(result == expected)

    # # test with time-delta type
    # data = numpy.array(data) * numpy.timedelta64(1, "s")
    #
    # # pandas
    # x = pandas.Series(data)
    # result = formula(x)
    # expected = pandas.Series(numpy.zeros_like(x), dtype=x.dtype)
    # assert all(result == expected)
    #
    # # numpy
    # x = numpy.array(data)
    # result = formula(x)
    # expected = numpy.zeros_like(x, dtype=x.dtype)
    # assert all(result == expected)


@mark.parametrize("formula", [true_like, lambda z: (z == z) | (z != z)])
def test_true_like(formula: SelfMap) -> None:
    """Analogous to `ones_like(x, dtype=bool)`.

    Candidates:
    - `where(x!=x, x!=x, x==x)`
    - `x==x | x!=x`
    """
    data = numpy.array([float("-inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")])
    x: Any
    result: Any
    expected: Any

    # torch
    x = torch.tensor(data)
    result = formula(x)
    expected = torch.ones_like(x, dtype=torch.bool)
    assert all(result == expected)

    # pandas
    x = pandas.Series(data)
    result = formula(x)
    expected = pandas.Series(numpy.ones_like(x, dtype=bool))
    assert all(result == expected)

    # numpy
    x = numpy.array(data)
    result = formula(x)
    expected = numpy.ones_like(x, dtype=numpy.bool_)
    assert all(result == expected)

    # test with time-delta type
    data = numpy.array(data) * numpy.timedelta64(1, "s")

    # pandas
    x = pandas.Series(data)
    result = formula(x)
    expected = pandas.Series(numpy.ones_like(x, dtype=bool))
    assert all(result == expected)

    # numpy
    x = numpy.array(data)
    result = formula(x)
    expected = numpy.ones_like(x, dtype=numpy.bool_)
    assert all(result == expected)


@mark.parametrize("formula", [false_like, lambda z: (z == z) ^ (z == z)])
def test_false_like(formula: SelfMap) -> None:
    """Analogous to `zeros_like(x, dtype=bool)`.

    Candidates:
    - XOR-trick: (x==x) ^ (x==x)
    - (x==x) & (x!=x)
    """
    data = numpy.array([float("-inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")])
    x: Any
    result: Any
    expected: Any

    # torch
    x = torch.tensor(data)
    result = formula(x)
    expected = torch.zeros_like(x, dtype=torch.bool)
    assert all(result == expected)

    # pandas
    x = pandas.Series(data)
    result = formula(x)
    expected = pandas.Series(numpy.zeros_like(x, dtype=bool))
    assert all(result == expected)

    # numpy
    x = numpy.array(data)
    result = formula(x)
    expected = numpy.zeros_like(x, dtype=numpy.bool_)
    assert all(result == expected)

    # test with time-delta type
    data = numpy.array(data) * numpy.timedelta64(1, "s")

    # pandas
    x = pandas.Series(data)
    result = formula(x)
    expected = pandas.Series(numpy.zeros_like(x, dtype=bool))
    assert all(result == expected)

    # numpy
    x = numpy.array(data)
    result = formula(x)
    expected = numpy.zeros_like(x, dtype=numpy.bool_)
    assert all(result == expected)


if __name__ == "__main__":
    pass
