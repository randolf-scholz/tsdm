r"""Test other protocols."""

from collections.abc import Iterable
from typing import (
    ClassVar,
    Protocol,
    TypeIs,
    _ProtocolMeta as ProtocolMeta,
    runtime_checkable,
)


class _SlottedMeta(ProtocolMeta):
    r"""Metaclass for `Slotted`.

    FIXME: https://github.com/python/cpython/issues/112319
    This issue will make the need for metaclass obsolete.
    """

    def __instancecheck__(cls, instance: object, /) -> TypeIs[Slotted]:  # noqa: N805
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass: type, /) -> TypeIs[type[Slotted]]:  # noqa: N805
        slots = getattr(subclass, "__slots__", None)
        return isinstance(slots, str | Iterable)


@runtime_checkable
class Slotted(Protocol, metaclass=_SlottedMeta):
    r"""Protocol for objects that are slotted."""

    __slots__: ClassVar[tuple[str, ...]] = ()


def is_slotted(obj: object, /) -> TypeIs[Slotted]:
    r"""Check if the object is slotted."""
    return hasattr(obj, "__slots__")


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
        case Slotted():  # pyrefly: ignore[unsafe-overlap]
            pass
        case _:
            raise AssertionError


def test_slotted_no_match() -> None:
    match NotSlotted(1, 2):
        case Slotted():
            raise AssertionError
        case _:
            pass  # type: ignore[unreachable]


def test_slotted_types() -> None:
    r"""Test the Slotted protocol."""
    _typ: type[Slotted] = MySlotted
    _obj: Slotted = MySlotted(1, 2)


def test_slotted_protocol() -> None:
    r"""Test the Slotted protocol."""
    assert isinstance(MySlotted(1, 2), Slotted)  # pyrefly: ignore[unsafe-overlap]
    assert issubclass(MySlotted, Slotted)  # type: ignore[arg-type]
    assert issubclass(Slotted, Slotted)  # type: ignore[arg-type]


def test_is_slotted() -> None:
    r"""Test the is_slotted utility."""
    # check an instance
    assert is_slotted(MySlotted(1, 2))
    assert isinstance(MySlotted(1, 2), Slotted)  # pyrefly: ignore[unsafe-overlap]
    # check the type
    assert is_slotted(MySlotted)
    assert issubclass(MySlotted, Slotted)  # type: ignore[arg-type]


def test_not_slotted() -> None:
    r"""Test the Slotted protocol."""
    # check an instance
    assert not is_slotted(NotSlotted(1, 2))
    assert not isinstance(NotSlotted(1, 2), Slotted)
    # check the type
    assert not is_slotted(NotSlotted)
    assert not issubclass(NotSlotted, Slotted)  # type: ignore[unreachable]
