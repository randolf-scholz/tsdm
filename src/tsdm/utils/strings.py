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
    "repr_short",
]
__ALL__ = dir() + __all__

import inspect
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import Field, is_dataclass
from functools import partial
from typing import Any, Final, Optional, overload

from pandas import DataFrame, Index, MultiIndex, Series
from torch import Tensor

from tsdm.utils.constants import BUILTIN_CONSTANTS
from tsdm.utils.types.dtypes import TYPESTRINGS, ScalarDType
from tsdm.utils.types.protocols import Array, Dataclass, NTuple

__logger__ = logging.getLogger(__name__)

MAXITEMS: Final[int] = 20
r"""Default maxitems for repr_funcs."""
MAXITEMS_INLINE: Final[int] = 5
r"""Default maxitems for repr_funcs."""
LINEBREAKS: Final[bool] = True
r"""Default linebreaks for repr_funcs."""
PADDING: Final[int] = 4
r"""Default padding for repr_funcs."""
RECURSIVE: Final[bool | int] = 1
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
    if isinstance(obj, str):
        return obj
    if inspect.isclass(obj) or inspect.isbuiltin(obj):
        return repr(obj)
    if is_dataclass(obj):
        return repr_dataclass(obj, **kwargs)
    if isinstance(obj, Array | Tensor | Series | DataFrame | Index):
        return repr_array(obj, **kwargs)
    if isinstance(obj, Mapping):  # type: ignore[unreachable]
        return repr_mapping(obj, **kwargs)
    if isinstance(obj, NTuple):
        return repr_namedtuple(obj, **kwargs)
    if isinstance(obj, Sequence):
        return repr_sequence(obj, **kwargs)
    if type(obj).__module__.split(".", maxsplit=1)[0] == "tsdm":
        return f"{obj.__class__.__name__}()"
    return repr(obj)


def repr_short(obj: Any, /, **_: Any) -> str:
    r"""Return a string representation of an object."""
    if isinstance(obj, str):
        return obj
    if inspect.isclass(obj) or inspect.isbuiltin(obj):
        return repr(obj)
    for item in BUILTIN_CONSTANTS:
        if obj is item:
            return repr(obj)
    if is_dataclass(obj):
        return repr_dataclass(obj, recursive=False)
    if isinstance(obj, NTuple):
        return repr_namedtuple(obj, recursive=False)
    if isinstance(obj, Array | DataFrame | Series):
        return repr_array(obj)
    if isinstance(obj, Mapping):  # type: ignore[unreachable]
        return repr_mapping(obj, recursive=False)
    if isinstance(obj, Sequence):
        return repr_sequence(obj, recursive=False)
    return repr(type(obj))


def repr_mapping(
    obj: Mapping,
    /,
    *,
    align: bool = ALIGN,
    identifier: Optional[str] = None,
    indent: int = 0,
    linebreaks: Optional[bool] = None,
    maxitems: Optional[int] = None,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a mapping object.

    - if recursive=True:  Name<Mapping>{key: repr_short(value), ...}
    - if recursive=False: Name<Mapping>{key: repr_object(value), ...}


    type only

         name<Mapping>[length=3]

    less less recursive:

        name<Mapping>(key1, key2)

    less recursive:

        name<Mapping>(
            key1: object<Mapping>(key1, key2, ...),
            key2: object<tuple>(item1, item2, ...),


    partial recursive:

        name<Mapping>(
            key1: object<Mapping>(key1: value1, key2: value2, ...),
            key2: object<tuple>(item1, item2, ...),


    fully recursive:

        name<Mapping>(
            key1: object<Mapping>(
                     key: value,
                  ),
            key2: object<tuple>(
                     key: value,
                  ),
    """
    # set linebreaks
    if linebreaks is None:
        linebreaks = bool(recursive)
    elif linebreaks and len(obj) < 1:
        linebreaks = False

    # set maxitems
    if maxitems is None:
        if linebreaks:
            maxitems = MAXITEMS
        else:
            maxitems = MAXITEMS_INLINE

    br = "\n" if linebreaks else ""
    # key_sep = ": "
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    # set max_key_length
    if align and linebreaks:
        max_key_length = max((len(str(key)) for key in obj), default=0)
    else:
        max_key_length = 0

    # check builtin
    if type(obj) == dict:  # pylint: disable=unidiomatic-typecheck
        if title is None:
            title = ""
        if identifier is None:
            identifier = dict.__name__

    # set title
    cls = type(obj)
    if title is None:
        title = cls.__name__

    # set identifier
    if identifier is None and title != cls.__name__:
        identifier = cls.__name__
    if identifier is None and title == cls.__name__:
        for base in cls.__bases__:
            if issubclass(base, Mapping):
                identifier = base.__name__
                break
        else:
            identifier = "Mapping"

    # # TODO: automatic linebreak detection if string length exceeds max_length
    # if recursive:
    #     if repr_fun not in RECURSIVE_REPR_FUNS:
    #         raise ValueError("Must use repr_short for recursive=True.")
    #
    #     to_string = partial(
    #         repr_fun,
    #         align=align,
    #         indent=indent + max_key_length,
    #         padding=padding,
    #         recursive=recursive if isinstance(recursive, bool) else recursive - 1,
    #         repr_fun=repr_fun,
    #     )
    #
    #     # def to_string(x: Any) -> str:
    #     #     return repr_fun(x).replace("\n", br + pad)
    #
    # else:
    #     to_string = partial(repr_short, indent=indent + max_key_length)

    r = recursive if isinstance(recursive, bool) else recursive - 1 or False
    to_string = partial(
        repr_fun,
        align=align,
        indent=indent + max_key_length,
        padding=padding,
        recursive=r,
        repr_fun=repr_fun,
    )

    items = [(str(key), to_string(value)) for key, value in obj.items()]

    # Assemble the string
    string = f"{title}<{identifier}>" + "(" + br
    if len(obj) <= maxitems:
        string += f"{sep}{br}".join(
            f"{pad}{key:<{max_key_length}}: {value}" for key, value in items
        )
    else:
        string += "".join(
            f"{pad}{key:<{max_key_length}}: {value}{sep}{br}"
            for key, value in items[: maxitems // 2]
        )
        string += f"{pad}...\n"
        string += f"{sep}{br}".join(
            f"{pad}{key:<{max_key_length}}: {value}"
            for key, value in items[-maxitems // 2 :]
        )
    string += br + ")"

    # add indent
    string = string.replace("\n", "\n" + " " * indent)

    config = {
        "object": type(obj).__name__,
        "size": len(obj),
        "align": align,
        "identifier": identifier,
        "indent": indent,
        "linebreaks": linebreaks,
        "maxitems": maxitems,
        "padding": padding,
        "recursive": recursive,
        "repr_fun": repr_fun,
        "title": title,
    }
    __logger__.debug("config=%s", config)
    return string


def repr_sequence(
    obj: Sequence,
    /,
    *,
    align: bool = ALIGN,
    identifier: Optional[str] = None,
    indent: int = 0,
    linebreaks: Optional[bool] = None,
    maxitems: Optional[int] = None,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a sequence object.

    - if recursive=True:  Name<Sequence>(repr_short(item1), ...)
    - if recursive=False: Name<Sequence>(repr_object(item1), ...)

    semi recursive

        name<Sequence>(
            object<Sequence>(item1, item2, ...),
            object<tuple>(item1, item2, ...),
            object<Mapping>(key1: value1, ...),
        )

    fully recursive:

        name<Sequence>(
            object<Sequence>(
                1,
                2,
            ),
            object<Mapping>(
                key: value,
            ),
        )
    """
    # set linebreaks
    if linebreaks is None:
        linebreaks = bool(recursive)
    elif linebreaks and len(obj) < 1:
        linebreaks = False

    # set maxitems
    if maxitems is None:
        if linebreaks:
            maxitems = MAXITEMS
        else:
            maxitems = MAXITEMS_INLINE

    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    # determine brackets
    if isinstance(obj, list):
        left, right = "[", "]"
    elif isinstance(obj, set):
        left, right = "{", "}"
    elif isinstance(obj, tuple):
        left, right = "(", ")"
    else:
        left, right = "(", ")"

    # check builtin
    for builtin in (list, tuple, set):
        if type(obj) == builtin:  # pylint: disable=unidiomatic-typecheck
            if title is None:
                title = ""
            if identifier is None:
                identifier = builtin.__name__

    # set title
    cls = type(obj)
    if title is None:
        title = cls.__name__

    # set identifier
    if identifier is None and title != cls.__name__:
        identifier = cls.__name__
    if identifier is None and title == cls.__name__:
        for base in cls.__bases__:
            if issubclass(base, Mapping):
                identifier = base.__name__
                break
        else:
            identifier = "Sequence"
    #
    # if recursive:
    #     if repr_fun not in RECURSIVE_REPR_FUNS:
    #         raise ValueError("Must use repr_short for recursive=True.")
    #
    #     to_string = partial(
    #         repr_fun,
    #         align=align,
    #         indent=indent + pad,
    #         padding=padding,
    #         recursive=recursive if isinstance(recursive, bool) else recursive - 1,
    #         repr_fun=repr_fun,
    #     )
    #
    #     # def to_string(x: Any) -> str:
    #     #     return repr_fun(x).replace("\n", br + pad)
    #
    # else:
    #     to_string = partial(repr_short, indent=indent + pad)

    r = recursive if isinstance(recursive, bool) else recursive - 1 or False
    to_string = partial(
        repr_fun,
        align=align,
        indent=indent + padding,
        padding=padding,
        recursive=r,
        repr_fun=repr_fun,
    )

    # Assemble the string
    string = f"{title}<{identifier}>" + left + br
    if len(obj) <= maxitems:
        string += f"{sep}{br}".join(f"{pad}{to_string(value)}" for value in obj)
    else:
        string += "".join(
            f"{pad}{to_string(value)}{sep}{br}" for value in obj[: maxitems // 2]
        )
        string += f"{pad}...{sep}" + br
        string += f"{sep}{br}".join(
            f"{pad}{to_string(value)}" for value in obj[-maxitems // 2 :]
        )
    # string = string[: -len(sep)]
    string += br + right

    # add indent
    string = string.replace("\n", "\n" + " " * indent)

    config = {
        "object": type(obj).__name__,
        "size": len(obj),
        "align": align,
        "identifier": identifier,
        "indent": indent,
        "linebreaks": linebreaks,
        "maxitems": maxitems,
        "padding": padding,
        "recursive": recursive,
        "repr_fun": repr_fun,
        "title": title,
    }
    __logger__.debug("config=%s", config)
    return string


def repr_dataclass(
    obj: object,
    /,
    *,
    align: bool = ALIGN,
    identifier: Optional[str] = None,
    indent: int = 0,
    linebreaks: Optional[bool] = None,
    maxitems: Optional[int] = None,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a dataclass object.

    - recursive=True:  Name<dataclass>(item1, item2, ...)
    - recursive=False: Name<dataclass>(item1=repr(item1), item2=repr(item2), ...)
    """
    assert is_dataclass(obj), f"Object {obj} is not a dataclass."
    assert isinstance(obj, Dataclass), f"Object {obj} is not a dataclass."
    fields: dict[str, Field] = obj.__dataclass_fields__

    cls = type(obj)
    title = cls.__name__ if title is None else title

    if identifier is None and title != cls.__name__:
        identifier = cls.__name__
    if identifier is None and title == cls.__name__:
        for base in cls.__bases__:
            if is_dataclass(base):
                identifier = base.__name__
                break
        else:
            identifier = "dataclass"

    if recursive:
        return repr_mapping(
            {key: getattr(obj, key) for key, field in fields.items() if field.repr},
            align=align,
            identifier=identifier,
            indent=indent,
            linebreaks=linebreaks,
            maxitems=maxitems,
            padding=padding,
            recursive=recursive,
            repr_fun=repr_fun,
            title=title,
        )
    return repr_sequence(
        [key for key, field in fields.items() if field.repr],
        align=align,
        identifier=identifier,
        indent=indent,
        linebreaks=linebreaks,
        maxitems=maxitems,
        padding=padding,
        recursive=recursive,
        repr_fun=repr_fun,
        title=title,
    )


def repr_namedtuple(
    obj: NTuple,
    /,
    *,
    align: bool = ALIGN,
    identifier: Optional[str] = None,
    indent: int = 0,
    linebreaks: Optional[bool] = None,
    maxitems: Optional[int] = None,
    padding: int = PADDING,
    recursive: bool | int = RECURSIVE,
    repr_fun: Callable[..., str] = repr_object,
    title: Optional[str] = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    - recursive=True:  Name<tuple>(item1, item2, ...)
    - recursive=False: Name<tuple>(item1=repr(item1), item2=repr(item2), ...)
    """
    assert isinstance(obj, tuple), f"Object {obj} is not a namedtuple."
    assert isinstance(obj, NTuple), f"Object {obj} is not a namedtuple."
    title = type(obj).__name__ if title is None else title
    identifier = "tuple" if identifier is None else identifier

    if recursive:
        return repr_mapping(
            obj._asdict(),
            align=align,
            identifier=identifier,
            indent=indent,
            linebreaks=linebreaks,
            maxitems=maxitems,
            padding=padding,
            recursive=recursive,
            repr_fun=repr_fun,
            title=title,
        )
    return repr_sequence(
        obj._fields,
        align=align,
        identifier=identifier,
        indent=indent,
        linebreaks=linebreaks,
        maxitems=maxitems,
        padding=padding,
        recursive=recursive,
        repr_fun=repr_fun,
        title=title,
    )


def repr_array(
    obj: Array | DataFrame | Series | Tensor,
    /,
    *,
    title: Optional[str] = None,
    **_: Any,
) -> str:
    r"""Return a string representation of an array object."""
    assert isinstance(
        obj, (Index, Array, DataFrame, Series)
    ), f"Object {obj=} is not an array, but {type(obj)=}."

    title = type(obj).__name__ if title is None else title

    string = title + "["
    string += str(tuple(obj.shape))

    if isinstance(obj, (DataFrame, MultiIndex)):
        dtypes = [repr_dtype(dtype) for dtype in obj.dtypes]
        string += ", " + repr_sequence(dtypes, linebreaks=False, maxitems=5)
    elif isinstance(obj, (Index, Series, Array)):
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


RECURSIVE_REPR_FUNS = [
    repr_object,
    repr_mapping,
    repr_sequence,
    repr_dataclass,
    repr_namedtuple,
]
