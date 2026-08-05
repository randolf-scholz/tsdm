r"""Test other protocols."""

from typing import NamedTuple, Self

from tsdm.types.namedtuple import NTuple, is_namedtuple


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


def test_ntuple_match() -> None:
    match MyNamedTuple(1, 2):
        case NTuple():  # pyrefly: ignore[unsafe-overlap]
            pass
        case _:
            raise AssertionError


def test_ntuple_no_match() -> None:
    match NotNamedTuple(1, 2):
        case NTuple():
            raise AssertionError
        case _:
            pass


def test_namedtuple_types() -> None:
    r"""Test the NTuple protocol."""
    _typ: type[NTuple] = MyNamedTuple  # pyrefly: ignore[bad-assignment]
    _obj: NTuple = MyNamedTuple(1, 2)  # pyrefly: ignore[bad-assignment]


def test_namedtuple_protocol() -> None:
    r"""Test the NTuple protocol."""
    # check an instance
    assert isinstance(MyNamedTuple(1, 2), tuple)
    assert isinstance(MyNamedTuple(1, 2), NTuple)  # pyrefly: ignore[unsafe-overlap]
    assert issubclass(MyNamedTuple, tuple)
    assert issubclass(MyNamedTuple, NTuple)  # type: ignore
    assert issubclass(NTuple, NTuple)  # type: ignore


def test_is_namedtuple() -> None:
    r"""Test the is_namedtuple utility."""
    # check an instance
    assert isinstance(MyNamedTuple(1, 2), tuple)
    assert isinstance(MyNamedTuple(1, 2), NTuple)  # pyrefly: ignore[unsafe-overlap]
    assert is_namedtuple(MyNamedTuple(1, 2))
    # check the type
    assert issubclass(MyNamedTuple, tuple)
    assert is_namedtuple(MyNamedTuple)
    assert issubclass(MyNamedTuple, NTuple)  # type: ignore


def test_not_namedtuple() -> None:
    r"""Test the NTuple protocol."""
    # check an instance
    assert isinstance(NotNamedTuple(1, 2), tuple)
    assert not is_namedtuple(NotNamedTuple(1, 2))
    assert not isinstance(NotNamedTuple(1, 2), NTuple)
    # check the type
    assert issubclass(NotNamedTuple, tuple)
    assert not is_namedtuple(NotNamedTuple)
    assert not issubclass(NotNamedTuple, NTuple)  # type: ignore
