r"""Test sequence like protocol."""

import collections
from collections import abc
from types import EllipsisType, NoneType, NotImplementedType
from typing import cast

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import torch

from tsdm.testing import assert_protocol
from tsdm.types.protocols import Array

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
    # abc.ByteString      : True,  # deprecated!
    abc.Callable        : False,  # type: ignore[dict-item]
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
    pa.Array        : False,  # lacks __contains__
    pa.ChunkedArray : False,  # lacks __contains__
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
def test_array_builtins(*, cls: type, expected: bool) -> None:
    assert_protocol(cls, Array, expected=expected)


def test_array_static() -> None:
    _: type[Array]
    # builtins
    # _ = bytes  # ❌ __contains__
    # _ = dict  # ❌ __getitem__
    _ = list
    _ = range
    # _ = str  # ❌ __contains__
    _ = tuple
    # collections.abc
    # _ = abc.Mapping  # __getitem__ does not support slicing
    # _ = abc.MutableMapping  # __getitem__ does not support slicing
    _ = abc.MutableSequence  # type: ignore[type-abstract]
    _ = abc.Sequence  # type: ignore[type-abstract]
    # collections
    # _ = collections.ChainMap  # __getitem__ does not support slicing
    # _ = collections.Counter  # __getitem__ does not support slicing
    # _ = collections.OrderedDict  # __getitem__ does not support slicing
    # _ = collections.UserDict  # __getitem__ does not support slicing
    _ = collections.UserList
    _ = collections.UserString
    # _ = collections.defaultdict  # __getitem__ does not support slicing
    # _ = collections.deque  # __getitem__ does not support slicing
    # 3rd party
    _ = np.ndarray
    _ = pa.Array
    _ = pa.ChunkedArray
    _ = pd.DataFrame
    _ = pd.Index
    _ = pd.Series
    _ = pl.Series
    _ = torch.Tensor  # pyright: ignore[reportAssignmentType]  (__contains__ bad type hint)
    # check


def test_instances_static() -> None:
    _: Array[int]
    _ = (1, 2)
    _ = cast(tuple[int, int], (1, 2))
    _ = cast(tuple[int, ...], (1, 2))
    _ = [1, 2]
    _ = range(2)


def test_array_collections_abc() -> None:
    for name in dir(collections):
        if name.startswith("_"):
            continue
        cls = getattr(collections, name)
        if isinstance(cls, type):
            print(f"{name}: {issubclass(cls, Array)}")
