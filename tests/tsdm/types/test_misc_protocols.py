r"""Test other protocols."""

from dataclasses import dataclass
from typing import NamedTuple, Self

import pytest

from tsdm.types.protocols import (
    Dataclass,
    NTuple,
    Slotted,
    is_dataclass,
    is_namedtuple,
    is_slotted,
)


@dataclass
class MyDataclass:
    r"""Dummy dataclass."""

    x: int
    y: int


class NotDataclass:
    r"""Dummy class that is not a dataclass."""

    x: int
    y: int

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class MyNamedTuple(NamedTuple):
    r"""Dummy `NamedTuple` class."""

    x: int
    y: int


class NotNamedTuple(tuple[int, int]):  # noqa: SLOT001
    r"""Dummy class that's a `tuple`, but not a `NamedTuple`."""

    x: int
    y: int

    def __new__(cls, x: int, y: int) -> Self:
        return super().__new__(cls, (x, y))

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class MySlotted:
    r"""Dummy class with `__slots__`."""

    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class NotSlotted:
    r"""Dummy class without `__slots__`."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def test_dataclass_match() -> None:
    match MyDataclass(1, 2):
        case Dataclass():
            pass
        case _:
            raise AssertionError


def test_dataclass_no_match() -> None:
    match NotDataclass(1, 2):
        case Dataclass():
            raise AssertionError
        case _:
            pass


def test_ntuple_match() -> None:
    match MyNamedTuple(1, 2):
        case NTuple():
            pass
        case _:
            raise AssertionError


def test_ntuple_no_match() -> None:
    match NotNamedTuple(1, 2):
        case NTuple():
            raise AssertionError
        case _:
            pass


def test_slotted_match() -> None:
    match MySlotted(1, 2):
        case Slotted():
            pass
        case _:
            raise AssertionError


@pytest.mark.xfail(reason="https://github.com/python/cpython/issues/112319.")
def test_slotted_no_match() -> None:
    match NotSlotted(1, 2):
        case Slotted():
            raise AssertionError
        case _:
            pass


def test_dataclass_types() -> None:
    r"""Test the Dataclass protocol."""
    _typ: type[Dataclass] = MyDataclass
    _obj: Dataclass = MyDataclass(1, 2)
    del _typ, _obj


def test_namedtuple_types() -> None:
    r"""Test the NTuple protocol."""
    _typ: type[NTuple] = MyNamedTuple
    _obj: NTuple = MyNamedTuple(1, 2)
    del _typ, _obj


def test_slotted_types() -> None:
    r"""Test the Slotted protocol."""
    _typ: type[Slotted] = MySlotted
    _obj: Slotted = MySlotted(1, 2)
    del _typ, _obj


def test_dataclass_protocol() -> None:
    r"""Test the Dataclass protocol."""
    assert isinstance(MyDataclass(1, 2), Dataclass)
    assert issubclass(MyDataclass, Dataclass)  # type: ignore[misc]
    assert issubclass(Dataclass, Dataclass)  # type: ignore[misc]


def test_slotted_protocol() -> None:
    r"""Test the Slotted protocol."""
    assert isinstance(MySlotted(1, 2), Slotted)
    assert issubclass(MySlotted, Slotted)  # pyright: ignore[reportGeneralTypeIssues]
    assert issubclass(Slotted, Slotted)  # pyright: ignore[reportGeneralTypeIssues]


def test_is_dataclass() -> None:
    r"""Check the is_dataclass utility."""
    assert is_dataclass(MyDataclass(1, 2))
    assert is_dataclass(MyDataclass)
    assert not is_dataclass(NotDataclass(1, 2))
    assert not is_dataclass(NotDataclass)


def test_is_namedtuple() -> None:
    r"""Test the is_namedtuple utility."""
    assert is_namedtuple(MyNamedTuple(1, 2))
    assert is_namedtuple(MyNamedTuple)
    assert not is_namedtuple(NotNamedTuple(1, 2))
    assert not is_namedtuple(NotNamedTuple)


def test_is_slotted() -> None:
    r"""Test the is_slotted utility."""
    assert is_slotted(MySlotted(1, 2))
    assert is_slotted(MySlotted)
    assert not is_slotted(NotDataclass(1, 2))
    assert not is_slotted(NotDataclass)


def test_not_dataclass() -> None:
    r"""Test the Dataclass protocol."""
    assert not isinstance(NotDataclass(1, 2), Dataclass)
    assert not issubclass(NotDataclass, Dataclass)  # type: ignore[misc]


def test_not_namedtuple() -> None:
    r"""Test the NTuple protocol."""
    assert isinstance(NotNamedTuple(1, 2), tuple)
    assert issubclass(NotNamedTuple, tuple)
    assert not isinstance(NotNamedTuple(1, 2), NTuple)
    assert not issubclass(NotNamedTuple, NTuple)  # type: ignore[misc]


@pytest.mark.xfail(reason="https://github.com/python/cpython/issues/112319.")
def test_not_slotted() -> None:
    r"""Test the Slotted protocol."""
    assert not isinstance(NotSlotted(1, 2), Slotted)
    assert not issubclass(NotSlotted, Slotted)  # pyright: ignore[reportGeneralTypeIssues]


def test_namedtuple_protocol() -> None:
    r"""Test the NTuple protocol."""
    assert isinstance(MyNamedTuple(1, 2), tuple)
    assert issubclass(MyNamedTuple, tuple)
    assert isinstance(MyNamedTuple(1, 2), NTuple)
    assert issubclass(MyNamedTuple, NTuple)  # type: ignore[misc]
