r"""Test other protocols."""

from tsdm.types.protocols import (
    Slotted,
    is_slotted,
)


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


def test_slotted_match() -> None:
    match MySlotted(1, 2):
        case Slotted():
            pass
        case _:
            raise AssertionError


def test_slotted_no_match() -> None:
    match NotSlotted(1, 2):
        case Slotted():
            raise AssertionError
        case _:
            pass


def test_slotted_types() -> None:
    r"""Test the Slotted protocol."""
    _typ: type[Slotted] = MySlotted
    _obj: Slotted = MySlotted(1, 2)


def test_slotted_protocol() -> None:
    r"""Test the Slotted protocol."""
    assert isinstance(MySlotted(1, 2), Slotted)
    assert issubclass(MySlotted, Slotted)  # pyright: ignore[reportGeneralTypeIssues]
    assert issubclass(Slotted, Slotted)  # pyright: ignore[reportGeneralTypeIssues]


def test_is_slotted() -> None:
    r"""Test the is_slotted utility."""
    # check an instance
    assert is_slotted(MySlotted(1, 2))
    assert isinstance(MySlotted(1, 2), Slotted)
    # check the type
    assert is_slotted(MySlotted)
    assert issubclass(MySlotted, Slotted)  # pyright: ignore[reportGeneralTypeIssues]


def test_not_slotted() -> None:
    r"""Test the Slotted protocol."""
    # check an instance
    assert not is_slotted(NotSlotted(1, 2))
    assert not isinstance(NotSlotted(1, 2), Slotted)
    # check the type
    assert not is_slotted(NotSlotted)
    assert not issubclass(NotSlotted, Slotted)  # type: ignore[unreachable]
