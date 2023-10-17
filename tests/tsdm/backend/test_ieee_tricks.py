"""This module contains tests for some tricks with tensor-types.

Often, we need some constant tensor that is the same shape/device as given tensor.
Typical values are zeros, ones, NaNs, ±inf, True, False, etc.

Sometimes we would like to keep NaNs.

There are some tricks to create such tensors in a backend-agnostic way, using IEEE arithmetic.

`is_nan` can be tested via `x != x`.
"""

from typing import TypeVar

import numpy
import pandas
import torch
from pytest import mark

from tsdm.backend.universal import false_like, true_like
from tsdm.types.callback_protocols import SelfMap

T = TypeVar("T", pandas.Series, numpy.ndarray, torch.Tensor)

DATA = [float("-inf"), -1.0, 0.0, 1.0, float("inf"), float("nan")]
TIME = numpy.array(DATA) * numpy.timedelta64(1, "s")


@mark.parametrize("formula", [lambda z: z**0], ids=["x**0"])
@mark.parametrize(
    "data, expected",
    [
        (torch.tensor(DATA), torch.ones_like(torch.tensor(DATA))),
        (pandas.Series(DATA), pandas.Series(numpy.ones_like(DATA))),
        (numpy.array(DATA), numpy.ones_like(DATA)),
    ],
    ids=["torch", "pandas", "numpy"],
)
def test_make_ones_like(data: T, expected: T, formula: SelfMap[T]) -> None:
    """Analogous to `ones_like`.

    Candidates for creating ones are:
    - `x**0`  gets rid of NANs
    - `x / x` keeps NANs, introduces NaNs for inf

    However, `x**0` is not implemented for time-delta types.
    What could work
    """
    result = formula(data)
    assert all(result == expected)


@mark.parametrize("formula", [lambda z: z**0 - z**0], ids=["x**0 - x**0"])
@mark.parametrize(
    "data, expected",
    [
        (torch.tensor(DATA), torch.zeros_like(torch.tensor(DATA))),
        (pandas.Series(DATA), pandas.Series(numpy.zeros_like(DATA))),
        (numpy.array(DATA), numpy.zeros_like(DATA)),
    ],
    ids=["torch", "pandas", "numpy"],
)
def test_zeros_like(data: T, expected: T, formula: SelfMap[T]) -> None:
    """Analogous to `zeros_like`.

    For creating zeros there are multiple good candidates:
    - `x - x`  # keeps NANs
    - `x**0 - x**0`  # gets rid of NANs
    - `0 * x`      # keeps NANs, introduces NaNs for inf
    - `0 ** x`  # doesn't for negatives, keeps NANs
    """
    result = formula(data)
    assert all(result == expected)


@mark.parametrize(
    "formula",
    [
        true_like,
        lambda z: (z == z) | (z != z),  # pylint: disable=comparison-with-itself
    ],
    ids=["true_like", "(x==x)|(x!=x)"],
)
@mark.parametrize(
    "data, expected",
    [
        (torch.tensor(DATA), torch.ones_like(torch.tensor(DATA), dtype=torch.bool)),
        (pandas.Series(DATA), pandas.Series(numpy.ones_like(DATA, dtype=bool))),
        (numpy.array(DATA), numpy.ones_like(DATA, dtype=numpy.bool_)),
        (pandas.Series(TIME), pandas.Series(numpy.ones_like(TIME, dtype=bool))),
        (numpy.array(TIME), numpy.ones_like(TIME, dtype=numpy.bool_)),
    ],
    ids=["torch", "pandas", "numpy", "pandas-timedelta", "numpy-timedelta"],
)
def test_true_like(data: T, expected: T, formula: SelfMap[T]) -> None:
    """Analogous to `ones_like(x, dtype=bool)`.

    Candidates:
    - `where(x!=x, x!=x, x==x)`
    - `x==x | x!=x`
    """
    result = formula(data)
    assert all(result == expected)


@mark.parametrize(
    "formula",
    [
        false_like,
        lambda z: (z == z) ^ (z == z),  # pylint: disable=comparison-with-itself
    ],
    ids=["false_like", "(x==x)^(x==x)"],
)
@mark.parametrize(
    "data, expected",
    [
        (torch.tensor(DATA), torch.zeros_like(torch.tensor(DATA), dtype=torch.bool)),
        (pandas.Series(DATA), pandas.Series(numpy.zeros_like(DATA, dtype=bool))),
        (numpy.array(DATA), numpy.zeros_like(DATA, dtype=numpy.bool_)),
        (pandas.Series(TIME), pandas.Series(numpy.zeros_like(TIME, dtype=bool))),
        (numpy.array(TIME), numpy.zeros_like(TIME, dtype=numpy.bool_)),
    ],
    ids=["torch", "pandas", "numpy", "pandas-timedelta", "numpy-timedelta"],
)
def test_false_like(data: T, expected: T, formula: SelfMap[T]) -> None:
    """Analogous to `zeros_like(x, dtype=bool)`.

    Candidates:
    - XOR-trick: (x==x) ^ (x==x)
    - (x==x) & (x!=x)
    """
    result = formula(data)
    assert all(result == expected)
