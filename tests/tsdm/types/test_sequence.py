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

from tsdm.types.protocols import Array

EXPECTED_BUILTINS: dict[type, bool] = {
    EllipsisType       : False,
    NoneType           : False,
    NotImplementedType : False,
    bool               : False,
    bytes              : True,
    complex            : False,
    dict               : True,
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
    abc.ByteString      : True,  # deprecated!
    abc.Callable        : False,
    abc.Collection      : False,
    abc.Container       : False,
    abc.Coroutine       : False,
    abc.Generator       : False,
    abc.Hashable        : False,
    abc.ItemsView       : False,
    abc.Iterable        : False,
    abc.Iterator        : False,
    abc.KeysView        : False,
    abc.Mapping         : True,
    abc.MappingView     : False,
    abc.MutableMapping  : True,
    abc.MutableSequence : True,
    abc.MutableSet      : False,
    abc.Reversible      : False,
    abc.Sequence        : True,
    abc.Set             : False,
    abc.Sized           : False,
    abc.ValuesView      : False,
}  # fmt: skip

EXTECTED_COLLECTIONS: dict[type, bool] = {
    collections.ChainMap    : True,
    collections.Counter     : True,
    collections.OrderedDict : True,
    collections.UserDict    : True,
    collections.UserList    : True,
    collections.UserString  : True,
    collections.defaultdict : True,
    collections.deque       : True,
}  # fmt: skip

EXPECTED_3RD_PARTY: dict[type, bool] = {
    np.ndarray      : True,
    pa.Array        : True,
    pa.ChunkedArray : True,
    pd.DataFrame    : True,
    pd.Index        : True,
    pd.Series       : True,
    pl.DataFrame    : False,  # __getitem__ incompatible!
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
def test_array_builtins(cls, expected):
    assert issubclass(cls, Array) is expected


def test_array_static():
    # builtins
    _: type[Array] = bytes
    _: type[Array] = dict
    _: type[Array] = list
    _: type[Array] = range
    _: type[Array] = str
    _: type[Array] = tuple
    # collections.abc
    _: type[Array] = abc.ByteString
    _: type[Array] = abc.Mapping
    _: type[Array] = abc.MutableMapping
    _: type[Array] = abc.MutableSequence
    _: type[Array] = abc.Sequence
    # collections
    _: type[Array] = collections.ChainMap
    _: type[Array] = collections.Counter
    _: type[Array] = collections.OrderedDict
    _: type[Array] = collections.UserDict
    _: type[Array] = collections.UserList
    _: type[Array] = collections.UserString
    _: type[Array] = collections.defaultdict
    _: type[Array] = collections.deque
    # 3rd party
    _: type[Array] = np.ndarray
    _: type[Array] = pa.Array
    _: type[Array] = pa.ChunkedArray
    _: type[Array] = pd.DataFrame
    _: type[Array] = pd.Index
    _: type[Array] = pd.Series
    _: type[Array] = pl.Series
    _: type[Array] = torch.Tensor


def test_array_collections_abc():
    for name in dir(collections):
        if name.startswith("_"):
            continue
        cls = getattr(collections, name)
        if isinstance(cls, type):
            print(f"{name}: {issubclass(cls, Array)}")
