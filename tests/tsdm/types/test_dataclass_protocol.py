r"""Test other protocols."""

from dataclasses import dataclass

from tsdm.types.protocols import (
    Dataclass,
    is_dataclass,
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


def test_dataclass_types() -> None:
    r"""Test the Dataclass protocol."""
    _typ: type[Dataclass] = MyDataclass
    _obj: Dataclass = MyDataclass(1, 2)


def test_dataclass_protocol() -> None:
    r"""Test the Dataclass protocol."""
    assert isinstance(MyDataclass(1, 2), Dataclass)
    assert issubclass(MyDataclass, Dataclass)  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
    assert issubclass(Dataclass, Dataclass)  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]


def test_is_dataclass() -> None:
    r"""Check the is_dataclass utility."""
    # check an instance
    assert is_dataclass(MyDataclass(1, 2))
    assert isinstance(MyDataclass(1, 2), Dataclass)
    # check the type
    assert is_dataclass(MyDataclass)
    assert issubclass(MyDataclass, Dataclass)  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]


def test_not_dataclass() -> None:
    r"""Test the Dataclass protocol."""
    # check an instance
    assert not is_dataclass(NotDataclass(1, 2))
    assert not isinstance(NotDataclass(1, 2), Dataclass)
    # check the type
    assert not is_dataclass(NotDataclass)
    assert not issubclass(NotDataclass, Dataclass)  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
