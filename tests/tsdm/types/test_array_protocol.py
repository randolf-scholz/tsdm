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
    abc.Callable        : False,  # type: ignore[dict-item]  # pyright: ignore[reportAssignmentType]
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
def test_satisfies_array_protocol(*, cls: type, expected: bool) -> None:
    assert_protocol(cls, Array, expected=expected)


def test_array_collections_abc() -> None:
    for name in dir(collections):
        if name.startswith("_"):
            continue
        cls = getattr(collections, name)
        if isinstance(cls, type):
            print(f"{name}: {issubclass(cls, Array)}")


def type_array_assignable() -> None:
    # builtins
    # _00: type[Array] = bytes  # ❌ __contains__
    # _01: type[Array] = dict  # ❌ __getitem__
    _02: type[Array] = list
    _03: type[Array] = range
    # _04: type[Array] = str  # ❌ __contains__
    _05: type[Array] = tuple
    # collections.abc
    # _06: type[Array] = abc.Mapping  # __getitem__ does not support slicing
    # _07: type[Array] = abc.MutableMapping  # __getitem__ does not support slicing
    _08: type[Array] = abc.MutableSequence  # type: ignore[type-abstract]
    _09: type[Array] = abc.Sequence  # type: ignore[type-abstract]
    # collections
    # _10: type[Array] = collections.ChainMap  # __getitem__ does not support slicing
    # _11: type[Array] = collections.Counter  # __getitem__ does not support slicing
    # _11: type[Array] = collections.OrderedDict  # __getitem__ does not support slicing
    # _12: type[Array] = collections.UserDict  # __getitem__ does not support slicing
    _13: type[Array] = collections.UserList
    _14: type[Array] = collections.UserString
    # _15: type[Array] = collections.defaultdict  # __getitem__ does not support slicing
    # _16: type[Array] = collections.deque  # __getitem__ does not support slicing
    # 3rd party
    _17: type[Array] = np.ndarray
    _18: type[Array] = pa.Array
    _19: type[Array] = pa.ChunkedArray
    _20: type[Array] = pd.DataFrame
    _21: type[Array] = pd.Index
    _22: type[Array] = pd.Series
    _23: type[Array] = pl.Series
    _24: type[Array] = torch.Tensor
    # check


def type_integer_array_assignable() -> None:
    _0: Array[int]
    _1: Array[int] = (1, 2)
    _2: Array[int] = tuple([1, 2])  # noqa: C409
    _3: Array[int] = [1, 2]
    _4: Array[int] = range(2)
