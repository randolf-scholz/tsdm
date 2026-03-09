r"""Tests for ShapeLike protocol."""

from collections.abc import Iterator
from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd
import torch


@runtime_checkable
class ShapeLike(Protocol):
    r"""Protocol for shapes, very similar to tuple, but without `__contains__`.

    Note:
        - tensorflow.TensorShape is not a tuple, but has a similar API.
        - pytorch.Size is a tuple.
        - numpy.ndarray.shape is a tuple.
        - pandas.Series.shape is a tuple.

    References:
        - https://github.com/python/typeshed/blob/main/stdlib/builtins.pyi
    """

    # unary operations
    def __hash__(self) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __getitem__(self, item: int, /) -> int: ...  # int <: SupportsIndex

    # binary operations
    # NOTE: Not returning Self, because that's how tuple works.
    def __eq__(self, other: object, /) -> bool: ...
    def __ne__(self, other: object, /) -> bool: ...
    def __lt__(self, other: Self | tuple, /) -> bool: ...
    def __le__(self, other: Self | tuple, /) -> bool: ...
    # arithmetic
    def __add__(self, other: Self | tuple, /) -> ShapeLike: ...


def test_shapelike_protocol() -> None:
    r"""Test the Shape protocol."""
    data = [1, 2, 3]
    torch_tensor: torch.Tensor = torch.tensor(data)
    numpy_ndarray: np.ndarray = np.array(data)
    pandas_series: pd.Series = pd.Series(data)
    pandas_index: pd.Index = pd.Index(data)

    x: ShapeLike = (1, 2, 3)
    y: ShapeLike = torch_tensor.shape
    z: ShapeLike = numpy_ndarray.shape
    w: ShapeLike = pandas_series.shape
    v: ShapeLike = pandas_index.shape
    assert isinstance(x, ShapeLike)
    assert isinstance(y, ShapeLike)
    assert isinstance(z, ShapeLike)
    assert isinstance(w, ShapeLike)
    assert isinstance(v, ShapeLike)
