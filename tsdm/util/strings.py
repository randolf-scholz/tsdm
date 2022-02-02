r"""Utility functions for string manipulation."""

from __future__ import annotations

__all__ = [
    # Functions
    "snake2camel",
    # "camel2snake",
    "repr_array",
    "repr_mapping",
    "repr_sequence",
    "tensor_info",
]

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, overload

from numpy.typing import ArrayLike
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
    maxitems: int = 6,
    repr_fun: Callable[..., str] = repr,
    linebreaks: bool = True,
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
    linebreaks: bool = True,

    Returns
    -------
    str
    """
    br = "\n" if linebreaks else ""
    padding = " " * pad * linebreaks
    max_key_length = max(len(str(key)) for key in obj.keys())
    items = list(obj.items())
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    def to_string(x: Any) -> str:
        return repr_fun(x).replace("\n", "\n" + padding)

    if len(obj) <= maxitems:
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}{br}"
            for key, value in items
        )
    else:
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}{br}"
            for key, value in items[: maxitems // 2]
        )
        string += f"{padding}...\n"
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}{br}"
            for key, value in items[-maxitems // 2 :]
        )
    string += ")"
    return string


def repr_sequence(
    obj: Sequence,
    pad: int = 2,
    maxitems: int = 6,
    repr_fun: Callable[..., str] = repr,
    linebreaks: bool = True,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a sequence object.

    Parameters
    ----------
    obj: Sequence
    pad: int
    maxitems: Optional[int] = 6
    repr_fun: Callable[..., str] = repr
    linebreaks: bool = True,
    title: Optional[str] = None,

    Returns
    -------
    str
    """
    br = "\n" if linebreaks else ""
    sep = "" if linebreaks else ", "
    padding = " " * pad * linebreaks
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    def to_string(x: Any) -> str:
        return repr_fun(x).replace("\n", br + padding)

    if maxitems is None or len(obj) <= 6:
        string += sep.join(f"{padding}{to_string(value)}{br}" for value in obj)
    else:
        string += sep.join(
            f"{padding}{to_string(value)}{br}" for value in obj[: maxitems // 2]
        )
        string += f"{padding}..." + br
        string += sep.join(
            f"{padding}{to_string(value)}{br}" for value in obj[-maxitems // 2 :]
        )
    string += ")"

    return string


def repr_array(obj: ArrayLike, title: Optional[str] = None) -> str:
    r"""Return a string representation of a array object.

    Parameters
    ----------
    obj: ArrayLike
    title: Optional[str] = None

    Returns
    -------
    str
    """
    title = type(obj).__name__ if title is None else title

    if hasattr(obj, "shape"):
        shape: Iterable[int] = obj.shape  # type: ignore[union-attr]
        return title + str(list(shape))
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return "[" + ", ".join(repr_array(x) for x in obj) + "]"
    return repr(obj)


def repr_object(obj: Any) -> str:
    r"""Return a string representation of an object.

    Parameters
    ----------
    obj: Any

    Returns
    -------
    str
    """
    if isinstance(obj, Tensor):
        return repr_array(obj)
    if isinstance(obj, Mapping):
        return repr_mapping(obj)
    if isinstance(obj, Sequence):
        return repr_sequence(obj)
    return str(obj)


def tensor_info(x: Tensor) -> str:
    r"""Print useful information about Tensor."""
    return f"{x.__class__.__name__}[{tuple(x.shape)}, {x.dtype}, {x.device.type}]"
