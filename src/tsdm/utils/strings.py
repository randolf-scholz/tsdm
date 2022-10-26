r"""Utility functions for string manipulation."""

# from __future__ import annotations
#
# from __future__ import annotations

__all__ = [
    # Functions
    "snake2camel",
    # "camel2snake",
    "dict2string",
    "tensor_info",
    # repr functions
    "repr_array",
    "repr_dataclass",
    "repr_dtype",
    "repr_mapping",
    "repr_namedtuple",
    "repr_object",
    "repr_sequence",
    "repr_sized",
    "repr_type",
]
__ALL__ = dir() + __all__


import builtins
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import is_dataclass
from typing import Any, Final, Optional, overload

from pandas import DataFrame, Series
from torch import Tensor

from tsdm.utils.types.dtypes import TYPESTRINGS, ScalarDType
from tsdm.utils.types.protocols import Array, Dataclass, NTuple

MAXITEMS: Final[int] = 7
r"""Default maxitems for repr_funcs."""
LINEBREAKS: Final[bool] = True
r"""Default linebreaks for repr_funcs."""
PADDING: Final[int] = 4
r"""Default padding for repr_funcs."""
RECURSIVE: Final[bool | int] = True
r"""Default recursive for repr_funcs."""
ALIGN: Final[bool] = True
r"""Default align for repr_mapping."""


def __dir__() -> list[str]:
    return __ALL__


@overload
def snake2camel(s: str) -> str:
    ...


@overload
def snake2camel(s: list[str]) -> list[str]:
    ...


@overload
def snake2camel(s: tuple[str, ...]) -> tuple[str, ...]:
    ...


def snake2camel(s):
    r"""Convert ``snake_case`` to ``CamelCase``."""
    if isinstance(s, tuple):
        return tuple(snake2camel(x) for x in s)

    if isinstance(s, Iterable) and not isinstance(s, str):
        return [snake2camel(x) for x in s]

    if isinstance(s, str):
        substrings = s.split("_")
        return "".join(s[0].capitalize() + s[1:] for s in substrings)

    raise TypeError(f"Type {type(s)} nor understood, expected string or iterable.")


def tensor_info(x: Tensor) -> str:
    r"""Print useful information about Tensor."""
    return f"{x.__class__.__name__}[{tuple(x.shape)}, {x.dtype}, {x.device.type}]"


def dict2string(d: dict[str, Any]) -> str:
    r"""Return pretty string representation of dictionary."""
    max_key_length = max((len(key) for key in d), default=0)
    pad = " " * 2

    string = "dict(" + "\n"

    for key, value in sorted(d.items()):
        string += f"\n{pad}{key:<{max_key_length}}: {repr(value)}"

    string += "\n)"
    return string


def repr_object(obj: Any, /, **kwargs: Any) -> str:
    r"""Return a string representation of an object."""
    if type(obj).__name__ in dir(builtins):
        return str(obj)
    if isinstance(obj, Tensor):
        return repr_array(obj, **kwargs)
    if isinstance(obj, Mapping):
        return repr_mapping(obj, **kwargs)
    if isinstance(obj, NTuple):
        return repr_namedtuple(obj, **kwargs)
    if isinstance(obj, Sequence):
        return repr_sequence(obj, **kwargs)
    if is_dataclass(obj):
        return repr_dataclass(obj, **kwargs)
    try:
        return repr(obj)
    # Fallback Option
    except Exception:
        return repr(type(obj))


def repr_type(obj: Any, /) -> str:
    r"""Return a string representation of an object."""
    if obj is None:
        return str(None)
    if obj is True:
        return str(True)
    if obj is False:
        return str(False)
    if isinstance(obj, Array | DataFrame | Series):
        return repr_array(obj)
    if isinstance(obj, Sized):  # type: ignore[unreachable]
        return repr_sized(obj)
    if isinstance(obj, type):
        return obj.__name__
    return obj.__class__.__name__ + "()"


def repr_mapping(
    obj: Mapping,
    /,
    *,
    align: bool = ALIGN,
    linebreaks: bool = LINEBREAKS,
    maxitems: int = MAXITEMS,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a mapping object."""
    br = "\n" if linebreaks else ""
    # key_sep = ": "
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    keys = [str(key) for key in obj.keys()]
    max_key_length = max(len(key) for key in keys) if align else 0

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
    /,
    *,
    linebreaks: bool = LINEBREAKS,
    maxitems: int = MAXITEMS,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a sequence object."""
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    if isinstance(obj, list):
        title, left, right = "", "[", "]"
    elif isinstance(obj, set):
        title, left, right = "", "{", "}"
    elif isinstance(obj, tuple):
        title, left, right = "", "(", ")"
    else:
        title = type(obj).__name__ if title is None else title
        left, right = "(", ")"

    string = title + left + br

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
    string = string[: -len(sep)]
    string += right

    return string


def repr_dataclass(
    obj: object,
    /,
    *,
    align: bool = ALIGN,
    linebreaks: bool = LINEBREAKS,
    maxitems: int = MAXITEMS,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    """Return a string representation of a dataclass object."""
    assert is_dataclass(obj), f"Object {obj} is not a dataclass."
    assert isinstance(obj, Dataclass), f"Object {obj} is not a dataclass."
    return repr_mapping(
        {key: getattr(obj, key) for key in obj.__dataclass_fields__},
        align=align,
        linebreaks=linebreaks,
        maxitems=maxitems,
        padding=padding,
        recursive=recursive if isinstance(recursive, bool) else recursive - 1,
        repr_fun=repr_fun,
        title=type(obj).__name__ if title is None else title,
    )


def repr_namedtuple(
    obj: NTuple,
    /,
    *,
    align: bool = ALIGN,
    linebreaks: bool = LINEBREAKS,
    maxitems: int = MAXITEMS,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a namedtuple object."""
    assert isinstance(obj, tuple), f"Object {obj} is not a namedtuple."
    assert isinstance(obj, NTuple), f"Object {obj} is not a namedtuple."
    return repr_mapping(
        obj._asdict(),
        align=align,
        linebreaks=linebreaks,
        maxitems=maxitems,
        padding=padding,
        recursive=recursive if isinstance(recursive, bool) else recursive - 1,
        repr_fun=repr_fun,
        title=type(obj).__name__ if title is None else title,
    )


def repr_array(
    obj: Array | DataFrame | Series, /, *, title: Optional[str] = None
) -> str:
    r"""Return a string representation of an array object."""
    assert isinstance(obj, Array | DataFrame)

    title = type(obj).__name__ if title is None else title

    string = title + "["
    string += str(tuple(obj.shape))

    if isinstance(obj, DataFrame):
        dtypes = [repr_dtype(dtype) for dtype in obj.dtypes]
        string += ", " + repr_sequence(dtypes, linebreaks=False)
    elif isinstance(obj, Array):
        string += ", " + repr_dtype(obj.dtype)
    else:
        raise TypeError(f"Cannot get dtype of {type(obj)}")
    if isinstance(obj, Tensor):
        string += f"@{obj.device}"

    string += "]"
    return string


def repr_sized(obj: Sized, /, *, title: Optional[str] = None) -> str:
    r"""Return a string representation of a sized object."""
    title = type(obj).__name__ if title is None else title
    string = title + "["
    string += str(len(obj))
    string += "]"
    return string


def repr_dtype(
    obj: str | ScalarDType,
    /,
) -> str:
    r"""Return a string representation of a dtype object."""
    if isinstance(obj, str):
        return obj
    if obj in TYPESTRINGS:
        return TYPESTRINGS[obj]
    return str(obj)
