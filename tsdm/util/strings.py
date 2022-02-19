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
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from typing import Any, Optional, Union, overload

from pandas import DataFrame, Series
from torch import Tensor

from tsdm.util.types.protocols import Array, NTuple

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


def repr_object(obj: Any, **kwargs: Any) -> str:
    r"""Return a string representation of an object.

    Parameters
    ----------
    obj: Any

    Returns
    -------
    str
    """
    if isinstance(obj, Tensor):
        return repr_array(obj, **kwargs)
    if isinstance(obj, Mapping):
        return repr_mapping(obj, **kwargs)
    if isinstance(obj, NTuple):
        return repr_namedtuple(obj, **kwargs)
    if isinstance(obj, Sequence):
        return repr_sequence(obj, **kwargs)
    return str(obj)


def repr_mapping(
    obj: Mapping,
    *,
    linebreaks: bool = True,
    maxitems: int = 6,
    padding: int = 4,
    recursive: Union[bool, int] = True,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a mapping object.

    Parameters
    ----------
    obj: Mapping
    linebreaks: bool, default True
    maxitems: int, default 6
    padding: int
    recursive: bool, default True
    repr_fun: Callable[..., str], default repr_object
    title: Optional[str], default None,

    Returns
    -------
    str
    """
    br = "\n" if linebreaks else ""
    # key_sep = ": "
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks
    max_key_length = 0  # max(len(str(key)) for key in obj.keys())
    items = list(obj.items())
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    # TODO: automatic linebreak detection if string length exceeds max_length

    def to_string(x: Any) -> str:
        if recursive:
            if isinstance(recursive, bool):
                return repr_fun(x).replace("\n", br + pad)
            return repr_fun(x, recursive=recursive - 1).replace("\n", br + pad)
        return repr_type(x)

    # keys = [str(key) for key in obj.keys()]
    # values: list[str] = [to_string(x) for x in obj.values()]

    if len(obj) <= maxitems:
        string += "".join(
            f"{pad}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items
        )
    else:
        string += "".join(
            f"{pad}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items[: maxitems // 2]
        )
        string += f"{pad}...\n"
        string += "".join(
            f"{pad}{str(key):<{max_key_length}}: {to_string(value)}{sep}{br}"
            for key, value in items[-maxitems // 2 :]
        )
    string += ")"
    return string


def repr_sequence(
    obj: Sequence,
    *,
    linebreaks: bool = True,
    maxitems: int = 6,
    padding: int = 4,
    recursive: Union[bool, int] = True,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a sequence object.

    Parameters
    ----------
    obj: Sequence
    linebreaks: bool, default True
    maxitems: int, default 6
    padding: int
    recursive: bool, default True
    repr_fun: Callable[..., str], default repr_object
    title: Optional[str], default None,

    Returns
    -------
    str
    """
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks
    title = type(obj).__name__ if title is None else title
    string = title + "(" + br

    def to_string(x: Any) -> str:
        if recursive:
            if isinstance(recursive, bool):
                return repr_fun(x).replace("\n", br + pad)
            return repr_fun(x, recursive=recursive - 1).replace("\n", br + pad)
        return repr_type(x)

    if len(obj) <= maxitems:
        string += "".join(f"{pad}{to_string(value)}{sep}{br}" for value in obj)
    else:
        string += "".join(
            f"{pad}{to_string(value)}{sep}{br}" for value in obj[: maxitems // 2]
        )
        string += f"{pad}..." + br
        string += "".join(
            f"{pad}{to_string(value)}{sep}{br}" for value in obj[-maxitems // 2 :]
        )
    string += ")"

    return string


def repr_namedtuple(
    obj: NTuple,
    *,
    linebreaks: bool = True,
    maxitems: int = 6,
    padding: int = 4,
    recursive: Union[bool, int] = True,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    Parameters
    ----------
    obj: tuple
    linebreaks: bool, default True
    maxitems: int, default 6
    padding: int
    recursive: bool | int, default True
    repr_fun: Callable[..., str], default repr_object
    title: Optional[str], default None,

    Returns
    -------
    str
    """
    title = type(obj).__name__ if title is None else title
    return repr_mapping(
        obj._asdict(),
        padding=padding,
        maxitems=maxitems,
        title=title,
        repr_fun=repr_fun,
        linebreaks=linebreaks,
        recursive=recursive,
    )


def repr_array(obj: Array, *, title: Optional[str] = None) -> str:
    r"""Return a string representation of a array object.

    Parameters
    ----------
    obj: ArrayLike
    title: Optional[str] = None

    Returns
    -------
    str
    """
    assert isinstance(obj, Array)

    title = type(obj).__name__ if title is None else title

    string = title + "["
    string += str(obj.shape)

    if hasattr(obj, "dtype"):
        string += f", dtype={str(obj.dtype)}"  # type: ignore[attr-defined]
    elif isinstance(obj, DataFrame):
        dtypes: Series = obj.dtypes
        if len(dtypes.unique()) == 1:
            string += f", dtype={str(dtypes[0])}"
        else:
            string += ", dtype=mixed"

    string += "]"
    return string


def repr_sized(obj: Sized, *, title: Optional[str] = None) -> str:
    r"""Return a string representation of a sized object.

    Parameters
    ----------
    obj: Sized
    title: Optional[str], default None

    Returns
    -------
    str
    """
    title = type(obj).__name__ if title is None else title
    string = title + "["
    string += str(len(obj))
    string += "]"
    return string


def repr_type(obj: Any) -> str:
    r"""Return a string representation of an object.

    Parameters
    ----------
    obj: Any

    Returns
    -------
    str
    """
    if isinstance(obj, Array):
        return repr_array(obj)
    if isinstance(obj, Sized):
        return repr_sized(obj)
    if isinstance(obj, type):
        return obj.__name__
    return obj.__class__.__name__ + "()"
