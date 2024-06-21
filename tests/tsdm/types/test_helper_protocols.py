r"""Test other protocols."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from typing_extensions import assert_type

from tsdm.types.protocols import (
    Seq,
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


def test_sequence_protocol() -> None:
    r"""Validate the SequenceProtocol class."""
    # down-casting
    d: Sequence[int] = [1, 2, 3]
    _: Seq[int] = d

    def down_cast(x: Sequence[T]) -> Seq[T]:
        return x

    assert_type(down_cast([1, 2, 3]), Seq[int])

    # checking list
    seq_list: Seq[int] = [1, 2, 3]
    assert isinstance(seq_list, Sequence)
    assert isinstance(seq_list, Seq)

    # check inference 2
    seq_tup: Seq[int] = (1, 2, 3)  # pyright: ignore[reportAssignmentType]
    assert isinstance(seq_tup, Sequence)
    assert isinstance(seq_tup, Seq)

    # check string
    seq_str: str = "foo"
    assert isinstance(seq_str, Sequence)


def test_seq_inference() -> None:
    r"""Test inference for seq-protocol."""

    def as_seq(x: Seq[T]) -> Seq[T]:
        return x

    var_tuple: tuple[int, ...] = (1, 2, 3)
    seq_tup = as_seq(var_tuple)  # pyright: ignore[reportArgumentType]
    assert_type(seq_tup, Seq[int])  # pyright: ignore[reportAssertTypeFailure]


def test_get_interscetion_indexable() -> None:
    containers = [
        list,
        tuple,
        pd.Series,
        pd.Index,
        np.ndarray,
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
