r"""Test sequence like protocol."""

import collections
from collections import abc
from types import EllipsisType, NoneType, NotImplementedType

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from tsdm.types.datasets import IndexableDataset

EXPECTED_BUILTINS: dict[type, bool] = {
    EllipsisType       : False,
    NoneType           : False,
    NotImplementedType : False,
    bool               : False,
    bytes              : False,
    complex            : False,
    dict               : False,
    float              : False,
    frozenset          : False,
    int                : False,
    list               : True,
    object             : False,
    range              : True,
    set                : False,
    slice              : False,
    str                : False,
    tuple              : True,
    type               : False,
}  # fmt: skip


EXPECTED_COLLECTIONS_ABC: dict[type, bool] = {
    abc.AsyncGenerator  : False,
    abc.AsyncIterable   : False,
    abc.AsyncIterator   : False,
    abc.Awaitable       : False,
    abc.Callable        : False,  # pyrefly: ignore[bad-assignment]
    abc.Collection      : False,
    abc.Container       : False,
    abc.Coroutine       : False,
    abc.Generator       : False,
    abc.Hashable        : False,
    abc.ItemsView       : False,
    abc.Iterable        : False,
    abc.Iterator        : False,
    abc.KeysView        : False,
    abc.Mapping         : False,
    abc.MappingView     : False,
    abc.MutableMapping  : False,
    abc.MutableSequence : True,
    abc.MutableSet      : False,
    abc.Reversible      : False,
    abc.Sequence        : True,
    abc.Set             : False,
    abc.Sized           : False,
    abc.ValuesView      : False,
}  # fmt: skip

EXTECTED_COLLECTIONS: dict[type, bool] = {
    collections.ChainMap    : False,
    collections.Counter     : False,
    collections.OrderedDict : False,
    collections.UserDict    : False,
    collections.UserList    : True,
    collections.UserString  : True,  # unwanted, but w/e
    collections.defaultdict : False,
    collections.deque       : True,  # lie, does not support slicing
}  # fmt: skip

EXPECTED_3RD_PARTY: dict[type, bool] = {
    np.ndarray      : True,
    pa.Array        : True,  # lacks __contains__
    pa.ChunkedArray : True,  # lacks __contains__
    pd.DataFrame    : True,
    pd.Index        : True,
    pd.Series       : True,
    pl.DataFrame    : True,  # white lie
    pl.Series       : True,
    torch.Tensor    : True,
}  # fmt: skip


@pytest.mark.parametrize(
    ("cls", "expected"),
    (
        EXPECTED_BUILTINS
        | EXTECTED_COLLECTIONS
        | EXPECTED_COLLECTIONS_ABC
        | EXPECTED_3RD_PARTY
    ).items(),
)
def test_satisfies_array_protocol(*, cls: type, expected: bool) -> None:
    if expected:
        assert issubclass(cls, IndexableDataset)
    else:
        assert not issubclass(cls, IndexableDataset)


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param([], id="empty-list"),
        pytest.param([1, 2], id="integer-list"),
        pytest.param([torch.tensor([1, 2]), torch.tensor([3, 4])], id="tensor-list"),
        pytest.param((1, 2), id="tuple"),
        pytest.param(range(2), id="range"),
        pytest.param(np.array([1, 2]), id="numpy-array"),
        pytest.param(pa.array([1, 2]), id="arrow-array"),
        pytest.param(pa.chunked_array([[1], [2]]), id="arrow-chunked-array"),
    ],
)
def test_indexable_dataset_accepts_instances(*, arg: object) -> None:
    assert isinstance(arg, IndexableDataset)


@pytest.mark.parametrize(
    "arg",
    [
        pytest.param("", id="empty-string"),
        pytest.param("abc", id="string"),
        pytest.param(b"", id="empty-bytes"),
        pytest.param(b"abc", id="bytes"),
        pytest.param(set(), id="empty-set"),
        pytest.param({1, 2}, id="integer-set"),
        pytest.param({}, id="empty-dict"),
        pytest.param({"sample": torch.tensor([1, 2])}, id="tensor-dict"),
        pytest.param({0: 1, 1: 2}, id="integer-dict"),
        pytest.param(collections.UserDict({0: 1}), id="user-dict"),
        pytest.param(collections.ChainMap({0: 1}), id="chain-map"),
    ],
)
def test_indexable_dataset_rejects_instances(*, arg: object) -> None:
    assert not isinstance(arg, IndexableDataset)


def check_indexable_dataset_assignable() -> None:
    _integers: IndexableDataset[int] = [1, 2]
    _tensors: IndexableDataset[torch.Tensor] = [torch.tensor([1, 2])]


def test_array_collections_abc() -> None:
    for name in dir(collections):
        if name.startswith("_"):
            continue
        cls = getattr(collections, name)
        if isinstance(cls, type):
            print(f"{name}: {issubclass(cls, IndexableDataset)}")


def check_array_assignable() -> None:
    # builtins
    # _00: type[Array] = bytes  # ❌ __contains__
    # _01: type[Array] = dict  # ❌ __getitem__
    _02: type[IndexableDataset] = list
    _03: type[IndexableDataset] = range
    # _04: type[Array] = str  # ❌ __contains__
    _05: type[IndexableDataset] = tuple
    # collections.abc
    # _06: type[Array] = abc.Mapping  # __getitem__ does not support slicing
    # _07: type[Array] = abc.MutableMapping  # __getitem__ does not support slicing
    _08: type[IndexableDataset] = abc.MutableSequence
    _09: type[IndexableDataset] = abc.Sequence
    # collections
    # _10: type[Array] = collections.ChainMap  # __getitem__ does not support slicing
    # _11: type[Array] = collections.Counter  # __getitem__ does not support slicing
    # _11: type[Array] = collections.OrderedDict  # __getitem__ does not support slicing
    # _12: type[Array] = collections.UserDict  # __getitem__ does not support slicing
    _13: type[IndexableDataset] = collections.UserList
    _14: type[IndexableDataset] = collections.UserString
    # _15: type[Array] = collections.defaultdict  # __getitem__ does not support slicing
    # _16: type[Array] = collections.deque  # __getitem__ does not support slicing
    # 3rd party
    _17: type[IndexableDataset] = np.ndarray
    _18: type[IndexableDataset] = pa.Array
    _19: type[IndexableDataset] = pa.ChunkedArray
    _20: type[IndexableDataset] = pd.DataFrame
    _21: type[IndexableDataset] = pd.Index
    _22: type[IndexableDataset] = pd.Series
    _23: type[IndexableDataset] = pl.Series
    _24: type[IndexableDataset] = torch.Tensor
    # check


def check_integer_array_assignable() -> None:
    _0: IndexableDataset[int]
    _1: IndexableDataset[int] = (1, 2)
    _2: IndexableDataset[int] = tuple([1, 2])  # ruff: ignore[C409]
    _3: IndexableDataset[int] = [1, 2]
    _4: IndexableDataset[int] = range(2)
