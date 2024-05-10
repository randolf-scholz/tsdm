r"""Test other protocols."""

from collections.abc import Mapping, Sequence

import numpy
import pandas
import torch
from numpy._typing import NDArray
from typing_extensions import assert_type

from tsdm.types.protocols import (
    SequenceProtocol,
    ShapeLike,
    SupportsKeysAndGetItem,
    SupportsKwargs,
)
from tsdm.types.variables import K, T, V


def test_supportskwargs() -> None:
    r"""Test the SupportsKwargs protocol."""

    class StrKeys:
        r"""Dummy class that supports `**kwargs`."""

        @staticmethod
        def keys() -> list[str]:
            return ["some", "strings"]

        def __getitem__(self, key: str) -> int:
            return len(key)

    class IntKeys:
        r"""Dummy class that does not support `**kwargs`."""

        @staticmethod
        def keys() -> list[int]:
            return [1, 2]

        def __getitem__(self, key: int) -> int:
            return key

    assert isinstance(StrKeys(), SupportsKeysAndGetItem)
    assert isinstance(IntKeys(), SupportsKeysAndGetItem)
    assert isinstance(StrKeys(), SupportsKwargs)
    assert not isinstance(IntKeys(), SupportsKwargs)


def test_shapelike_protocol() -> None:
    r"""Test the Shape protocol."""
    data = [1, 2, 3]
    torch_tensor: torch.Tensor = torch.tensor(data)
    numpy_ndarray: NDArray = numpy.array(data)
    pandas_series: pandas.Series = pandas.Series(data)
    pandas_index: pandas.Index = pandas.Index(data)

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


def test_sequence_protocol() -> None:
    r"""Validate the SequenceProtocol class."""

    def foo(x: Sequence[T]) -> SequenceProtocol[T]:
        return x

    # checking list
    seq_list: SequenceProtocol[int] = [1, 2, 3]
    assert isinstance(seq_list, Sequence)
    assert isinstance(seq_list, SequenceProtocol)

    # check tuple
    seq_tup: SequenceProtocol[int] = (1, 2, 3)
    assert isinstance(seq_tup, Sequence)
    assert isinstance(seq_tup, SequenceProtocol)

    # check string
    seq_str: str = "foo"
    assert isinstance(seq_str, Sequence)


def test_get_interscetion_indexable() -> None:
    containers = [
        list,
        tuple,
        pandas.Series,
        pandas.Index,
        numpy.ndarray,
        torch.Tensor,
    ]
    shared_attrs = set.intersection(*[set(dir(c)) for c in containers])
    excluded_attrs = {
        "__init__",
        "__getattribute__",
        "__sizeof__",
        "__init_subclass__",
        "__subclasshook__",
        "__getstate__",
        "__dir__",
        "__doc__",
        "__delattr__",
        "__class__",
        "__format__",
        "__reduce_ex__",
        "__setattr__",
        "__repr__",
        "__str__",
        "__reduce__",
    }
    attrs = sorted(shared_attrs - excluded_attrs)
    print("Shared attributes:\n" + "\n".join(attrs))


def test_supportskeysgetitem() -> None:
    r"""Test the SupportsKeysAndGetItem protocol."""

    def foo(x: Mapping[K, V]) -> SupportsKeysAndGetItem[K, V]:
        return x

    assert_type(foo({"a": 1}), SupportsKeysAndGetItem[str, int])
