r"""Utilities for typing context."""

__all__ = [
    "is_classvar",
]

from typing import ClassVar, ForwardRef


# FIXME: Use TypeForm with typing_extensions==4.13.0
def is_classvar(tp: object) -> bool:
    r"""Check if the type annotation is a ClassVar."""
    if tp is ClassVar:
        return True
    if isinstance(tp, str | ForwardRef):
        # TODO: add support for ForwardRef
        raise NotImplementedError("ForwardRef / string annotation is not supported.")
    if (origin := getattr(tp, "__origin__", None)) is not None:
        return is_classvar(origin)
    return False
