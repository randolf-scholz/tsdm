r"""Utility functions for string manipulation."""

from __future__ import annotations

__all__ = [
    # Functions
    "snake2camel",
    # "camel2snake",
    "repr_mapping",
    "repr_sequence",
    "tensor_info",
]

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, overload

from torch import Tensor

__logger__ = logging.getLogger(__name__)


@overload
def snake2camel(s: str) -> str:  # type: ignore[misc]
    ...


@overload
def snake2camel(s: Iterable[str]) -> list[str]:
    ...


def snake2camel(s):
    """Convert ``snake_case`` to ``CamelCase``.

    Parameters
    ----------
    s: str | Iterable[str]

    Returns
    -------
    str | Iterable[str]
    """
    if isinstance(s, Iterable) and not isinstance(s, str):
        return [snake2camel(x) for x in s]

    substrings = s.split("_")
    return "".join(s[0].capitalize() + s[1:] for s in substrings)


def repr_mapping(
    obj: Mapping,
    pad: int = 2,
    maxitems: Optional[int] = 6,
    repr_fun: Callable[..., str] = repr,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a mapping object.

    Parameters
    ----------
    obj: Mapping
    pad: int
    maxitems: Optional[int] = 6
    repr_fun: Callable[..., str] = repr
    title: Optional[str] = repr,

    Returns
    -------
    str
    """
    padding = " " * pad

    def to_string(x: Any) -> str:
        return repr_fun(x).replace("\n", "\n" + padding)

    max_key_length = max(len(str(key)) for key in obj.keys())
    items = list(obj.items())
    if title is None:
        title = type(obj).__name__
    string = title + "(\n"

    if maxitems is None or len(obj) <= maxitems:
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}\n"
            for key, value in items
        )
    else:
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}\n"
            for key, value in items[: maxitems // 2]
        )
        string += f"{padding}...\n"
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}\n"
            for key, value in items[-maxitems // 2 :]
        )
    string += ")"
    return string


def repr_sequence(
    obj: Sequence,
    pad: int = 2,
    maxitems: Optional[int] = 6,
    repr_fun: Callable[..., str] = repr,
) -> str:
    r"""Return a string representation of a sequence object.

    Parameters
    ----------
    obj: Sequence
    pad: int
    maxitems: Optional[int] = 6
    repr_fun: Callable[..., str] = repr

    Returns
    -------
    str
    """
    padding = " " * pad

    def to_string(x: Any) -> str:
        return repr_fun(x).replace("\n", "\n" + padding)

    if maxitems is None:
        maxitems = len(obj)

    string = type(obj).__name__ + "(\n"

    if maxitems is None or len(obj) <= 6:
        string += "".join(f"{padding}{to_string(value)}\n" for value in obj)
    else:
        string += "".join(
            f"{padding}{to_string(value)}\n" for value in obj[: maxitems // 2]
        )
        string += f"{padding}...\n"
        string += "".join(
            f"{padding}{to_string(value)}\n" for value in obj[-maxitems // 2 :]
        )
    string += ")"

    return string


def tensor_info(x: Tensor) -> str:
    r"""Print useful information about Tensor."""
    return f"{x.__class__.__name__}[{tuple(x.shape)}, {x.dtype}, {x.device.type}]"
