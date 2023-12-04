r"""Test other protocols."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy
import pandas
import torch
from numpy._typing import NDArray
from pytest import mark

from tsdm.types.protocols import (
    Dataclass,
    NTuple,
    SequenceProtocol,
    ShapeLike,
    SupportsKeysAndGetItem,
    SupportsKwargs,
    is_dataclass,
    is_namedtuple,
)
from tsdm.types.variables import any_var as T


def test_shapelike_protocol() -> None:
    """Test the Shape protocol."""
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


def test_dataclass_protocol() -> None:
    """Test the Dataclass protocol."""

    @dataclass
    class MyDataClass:
        """Dummy dataclass."""

        x: int
        y: int

    class Bar:
        """Dummy class."""

        x: int
        y: int

        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

    # type checking
    _typ: type[Dataclass] = MyDataClass
    _obj: Dataclass = MyDataClass(1, 2)
    del _typ, _obj

    foo = MyDataClass(1, 2)
    bar = Bar(1, 2)

    # test match-case
    match foo:
        case Dataclass():
            pass
        case _:
            raise AssertionError

    match foo:
        case MyDataClass(1, 2):
            pass
        case _:
            raise AssertionError

    match bar:
        case Dataclass():
            raise AssertionError
        case _:
            pass

    # check that Foo has __dataclass_fields__
    extra_attrs = set(dir(MyDataClass)) - set(dir(Bar))
    print(f"Additional attributes: {extra_attrs}")
    assert "__dataclass_fields__" in extra_attrs
    assert isinstance(MyDataClass.__dataclass_fields__, dict)

    # check that foo has __dataclass_fields__
    extra_attrs = set(dir(foo)) - set(dir(bar))
    print(f"Additional attributes: {extra_attrs}")
    assert "__dataclass_fields__" in extra_attrs
    assert isinstance(foo.__dataclass_fields__, dict)

    # check is_dataclass utility
    assert is_dataclass(foo)
    assert is_dataclass(MyDataClass)
    assert not is_dataclass(bar)
    assert not is_dataclass(Bar)

    # check that Foo is dataclass
    assert isinstance(foo, Dataclass)
    assert issubclass(MyDataClass, Dataclass)  # type: ignore[misc]
    assert not isinstance(bar, Dataclass)
    assert not issubclass(Bar, Dataclass)  # type: ignore[misc]


@mark.xfail(reason="Attribute __dataclass_fields__ does not exist on class.")
def test_dataclass_protocol_itself() -> None:
    assert issubclass(Dataclass, Dataclass)  # type: ignore[misc]


def test_namedtuple_protocol_itself() -> None:
    assert issubclass(NTuple, NTuple)  # type: ignore[misc]


def test_namedtuple_protocol() -> None:
    """Test the NTuple protocol."""

    class MyTuple(NamedTuple):
        """A point in 2D space."""

        x: int
        y: int

    class Bar(tuple[int, int]):
        """A point in 2D space."""

        x: int
        y: int

    # type checking
    _typ: type[NTuple] = MyTuple
    _obj: NTuple = MyTuple(1, 2)
    del _typ, _obj

    foo = MyTuple(1, 2)
    bar = Bar((1, 2))

    # test match-case compatibility
    match foo:
        case NTuple():
            pass
        case _:
            raise AssertionError

    match foo:
        case MyTuple(1, 2):
            pass
        case _:
            raise AssertionError

    match bar:
        case NTuple():
            raise AssertionError
        case _:
            pass

    # check that Foo has _fields
    extra_attrs = set(dir(MyTuple)) - set(dir(Bar))
    print(f"Additional attributes: {extra_attrs}")
    assert "_fields" in extra_attrs
    assert isinstance(MyTuple._fields, tuple)

    # check that foo has _fields
    extra_attrs = set(dir(foo)) - set(dir(bar))
    print(f"Additional attributes: {extra_attrs}")
    assert "_fields" in extra_attrs
    assert isinstance(foo._fields, tuple)

    # check is_namedtuple utility
    assert is_namedtuple(foo)
    assert is_namedtuple(MyTuple)
    assert not is_namedtuple(bar)
    assert not is_namedtuple(Bar)

    # check that point is tuple
    assert isinstance(foo, tuple)
    assert issubclass(MyTuple, tuple)
    assert not issubclass(tuple, MyTuple)

    # check that point is namedtuple
    assert isinstance(foo, NTuple)
    assert issubclass(MyTuple, NTuple)  # type: ignore[misc]
    assert not issubclass(NTuple, MyTuple)

    # check that tuple is not point
    assert not isinstance(bar, NTuple)
    assert not isinstance(bar, MyTuple)


def test_supportskwargs() -> None:
    class Foo:
        r"""Dummy class that supports `**kwargs`."""

        @staticmethod
        def keys() -> list[str]:
            return ["some", "strings"]

        def __getitem__(self, key: str) -> int:
            return len(key)

    class Bar:
        r"""Dummy class that does not support `**kwargs`."""

        @staticmethod
        def keys() -> list[int]:
            return [1, 2]

        def __getitem__(self, key: int) -> int:
            return key

    assert isinstance(Foo(), SupportsKeysAndGetItem)
    assert isinstance(Bar(), SupportsKeysAndGetItem)
    assert isinstance(Foo(), SupportsKwargs)
    assert not isinstance(Bar(), SupportsKwargs)


def test_sequence_protocol() -> None:
    """Validate the SequenceProtocol class."""

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
