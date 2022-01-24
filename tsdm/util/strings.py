r"""Utility functions for string manipulation."""

__all__ = [
    # Functions
    "snake2camel",
    # "camel2snake",
    "repr_mapping",
    "repr_sequence",
]

import logging
from collections.abc import Iterable, Mapping, Sequence, Callable
from typing import Optional, overload

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
    repr_fun: Optional[Callable[..., str]] = None,
) -> str:
    """Return a string representation of a mapping object.

    Parameters
    ----------
    obj: Mapping
    pad: int
    maxitems: Optional[int] = 6

    Returns
    -------
    str
    """
    to_string = repr if repr_fun is None else repr_fun

    padding = " " * pad

    max_key_length = max(len(str(key)) for key in obj.keys())
    items = list(obj.items())

    string = type(obj).__name__ + "(\n"

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
    repr_fun: Optional[Callable[..., str]] = None,
) -> str:
    """Return a string representation of a sequence object.

    Parameters
    ----------
    obj: Sequence
    pad: int

    Returns
    -------
    str
    """
    to_string = repr if repr_fun is None else repr_fun
    padding = " " * pad
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
