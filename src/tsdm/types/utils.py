r"""Utilities for typing context."""

__all__ = [
    "assert_protocol",
    "is_classvar",
]

from typing import ClassVar, ForwardRef, get_protocol_members, is_protocol

from typing_extensions import TypeForm


def is_classvar(tp: TypeForm, /) -> bool:
    r"""Check if the type annotation is a ClassVar."""
    if tp is ClassVar:
        return True
    if isinstance(tp, str | ForwardRef):
        # TODO: add support for ForwardRef
        raise NotImplementedError("ForwardRef / string annotation is not supported.")
    if (origin := getattr(tp, "__origin__", None)) is not None:
        return is_classvar(origin)
    return False


def assert_protocol(obj: object, proto: type, /) -> None:
    r"""Assert that the object is a given protocol."""
    if not is_protocol(proto):
        raise TypeError(f"{proto} is not a protocol!")

    if isinstance(obj, type):
        match = issubclass(obj, proto)
        name = obj.__name__
    else:
        match = isinstance(obj, proto)
        name = obj.__class__.__name__

    member = "a subtype" if isinstance(obj, type) else "an instance"
    msg = f"{name!r} is not a {member} of {proto.__name__!r}!"
    missing_attrs = sorted(get_protocol_members(proto) - set(dir(obj)))

    if not match:
        raise AssertionError(f"{msg}\n Missing Attributes: {missing_attrs}")
