r"""Test other protocols."""

from collections.abc import Sequence
from typing import assert_type

import numpy as np
import pandas as pd
import torch

from tsdm.types.abc import Seq


def test_sequence_protocol() -> None:
    r"""Validate the SequenceProtocol class."""
    # down-casting
    d: Sequence[int] = [1, 2, 3]
    _: Seq[int] = d

    tup: tuple[int, ...] = tuple(d)
    lst: list[int] = [1, 2, 3]

    def down_cast[T](x: Sequence[T]) -> Seq[T]:
        return x

    assert_type(down_cast([1, 2, 3]), Seq[int])

    # check list
    seq_list: Seq[int] = lst
    assert isinstance(seq_list, Sequence)
    assert isinstance(seq_list, Seq)

    # check tuple
    seq_tup: Seq[int] = tup
    assert isinstance(seq_tup, Sequence)
    assert isinstance(seq_tup, Seq)

    # check string
    seq_str: str = "foo"
    assert isinstance(seq_str, Sequence)


def test_seq_inference() -> None:
    r"""Test inference for seq-protocol."""

    def as_seq[T](x: Seq[T]) -> Seq[T]:
        return x

    var_tuple: tuple[int, ...] = (1, 2, 3)
    seq_tup = as_seq(var_tuple)  # pyright: ignore[reportArgumentType]
    assert_type(seq_tup, Seq[int])  # pyright: ignore[reportAssertTypeFailure]


def show_interscetion_indexable_types() -> None:
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
