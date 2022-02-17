r"""Utility functions for string manipulation."""

from __future__ import annotations

__all__ = [
    # Functions
    "snake2camel",
    # "camel2snake",
    "repr_array",
    "repr_mapping",
    "repr_sequence",
    "repr_namedtuple",
    "tensor_info",
    "dict2string",
]

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, NamedTuple, Optional, overload

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


def tensor_info(x: Tensor) -> str:
    r"""Print useful information about Tensor."""
    return f"{x.__class__.__name__}[{tuple(x.shape)}, {x.dtype}, {x.device.type}]"


def dict2string(d: dict[str, Any]) -> str:
    r"""Return pretty string representation of dictionary.

    Vertically aligns keys.

    Parameters
    ----------
    d: dict[str, Any]

    Returns
    -------
    str
    """
    max_key_length = max((len(key) for key in d), default=0)
    pad = " " * 2

    string = "dict(" + "\n"

    for key, value in sorted(d.items()):
        string += f"\n{pad}{key:<{max_key_length}}: {repr(value)}"

    string += "\n)"
    return string


def repr_mapping(
    obj: Mapping,
    pad: int = 4,
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
    sep = "," if linebreaks else ", "
    padding = " " * pad * linebreaks
    max_key_length = max(len(str(key)) for key in obj.keys())
    items = list(obj.items())
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    def to_string(x: Any) -> str:
        return repr_fun(x).replace("\n", "\n" + padding)

    if len(obj) <= maxitems:
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items
        )
    else:
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items[: maxitems // 2]
        )
        string += f"{padding}...\n"
        string += "".join(
            f"{padding}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items[-maxitems // 2 :]
        )
    string += ")"
    return string


def repr_sequence(
    obj: Sequence,
    pad: int = 4,
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
    sep = "," if linebreaks else ", "
    padding = " " * pad * linebreaks
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    def to_string(x: Any) -> str:
        return repr_fun(x).replace("\n", br + padding)

    if maxitems is None or len(obj) <= 6:
        string += "".join(f"{padding}{to_string(value)}{sep}{br}" for value in obj)
    else:
        string += "".join(
            f"{padding}{to_string(value)}{sep}{br}" for value in obj[: maxitems // 2]
        )
        string += f"{padding}..." + br
        string += "".join(
            f"{padding}{to_string(value)}{sep}{br}" for value in obj[-maxitems // 2 :]
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
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return repr_namedtuple(obj)
    if isinstance(obj, Sequence):
        return repr_sequence(obj)
    return str(obj)


def repr_namedtuple(
    obj: tuple,
    padding: int = 4,
    maxitems: int = 6,
    repr_fun: Callable[..., str] = repr,
    linebreaks: bool = True,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    Parameters
    ----------
    obj
    padding
    maxitems
    repr_fun
    linebreaks
    title

    Returns
    -------
    str
    """
    assert hasattr(obj, "_fields")
    keys = obj._fields  # type: ignore[attr-defined]

    name = obj.__class__.__name__
    pad = " " * padding
    string = f"{name}("

    strings: list[str] = []
    max_key_len = max(len(key) for key in keys)
    objs = [repr(x).replace("\n", "\n" + " " * max_key_len) for x in obj]

    for k, x in zip(keys, objs):
        strings.append(f"{k:<{max_key_len}} = {x}")

    total_length = len("".join(strings))

    if total_length < 80:
        string += ", ".join(strings)
    else:
        string += "".join(f"\n{pad}{s}," for s in strings)
        string += "\n"
    string += ")"

    return string
