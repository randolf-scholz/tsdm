r"""Utility functions for string manipulation."""

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

import inspect
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import is_dataclass
from functools import partial
from types import FunctionType
from typing import Any, Final, Optional, Protocol, cast, overload

from pandas import DataFrame, Index, MultiIndex, Series
from torch import Tensor

from tsdm.types.aliases import ScalarDType
from tsdm.types.dtypes import TYPESTRINGS
from tsdm.types.protocols import Array, Dataclass, NTuple
from tsdm.utils.constants import BUILTIN_CONSTANTS, BUILTIN_TYPES

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
def snake2camel(s: str) -> str: ...
@overload
def snake2camel(s: list[str]) -> list[str]: ...
@overload
def snake2camel(s: tuple[str, ...]) -> tuple[str, ...]: ...
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


class ReprProtocol(Protocol):
    """Protocol for recursive repr functions."""

    def __call__(
        self,
        obj: Any,
        /,
        *,
        align: bool = ALIGN,
        identifier: Optional[str] = None,
        indent: int = 0,
        linebreaks: Optional[bool] = None,
        maxitems: Optional[int] = None,
        padding: int = PADDING,
        recursive: bool | int = RECURSIVE,
        repr_fun: Callable[..., str] = NotImplemented,
        title: Optional[str] = None,
        wrapped: Optional[object] = None,
    ) -> str: ...


def get_identifier(obj: Any, /, **_: Any) -> str:
    r"""Return the identifier of an object."""
    if is_dataclass(obj):
        identifier = "<dataclass>"
    elif isinstance(obj, tuple):
        identifier = "<tuple>"
    elif isinstance(obj, Array | DataFrame | Series):
        identifier = "<array>"
    elif isinstance(obj, Mapping):  # type: ignore[unreachable]
        identifier = "<mapping>"
    elif isinstance(obj, Sequence):
        identifier = "<sequence>"
    elif isinstance(obj, type):
        identifier = "<type>"
    elif isinstance(obj, FunctionType):
        identifier = "<function>"
    else:
        identifier = ""
    if type(obj) in BUILTIN_TYPES:
        identifier = ""
    return identifier


# def get_identifier_cls(cls: type, /, **_: Any) -> str:
#     r"""Return the identifier of an object."""
#     if is_dataclass(cls):
#         identifier = "<dataclass>"
#     elif issubclass(cls, NTuple):
#         identifier = "<tuple>"
#     elif issubclass(cls, Array | DataFrame | Series):
#         identifier = "<array>"
#     elif issubclass(cls, Mapping):
#         identifier = "<mapping>"
#     elif issubclass(cls, Sequence):
#         identifier = "<sequence>"
#     elif issubclass(cls, type):
#         identifier = "<type>"
#     elif issubclass(cls, FunctionType):
#         identifier = "<function>"
#     else:
#         identifier = ""
#     if cls in BUILTIN_TYPES:
#         identifier = ""
#     return identifier


def repr_object(
    obj: Any, /, *, fallback: Callable[..., str] = repr, **kwargs: Any
) -> str:
    r"""Return a string representation of an object.

    Special casing for a bunch of cases.
    """
    __logger__.debug("repr_object: %s, %s", type(obj), kwargs)

    x = kwargs.get("wrapped", obj)
    x = obj if x is None else obj

    if isinstance(x, str):
        __logger__.debug("repr_object: → str")
        return obj
    if inspect.isclass(x) or inspect.isbuiltin(x):
        __logger__.debug("repr_object: → type")
        return repr(obj)
    if is_dataclass(x):
        __logger__.debug("repr_object: → dataclass")
        return repr_dataclass(obj, **kwargs)
    if isinstance(x, Array | Tensor | Series | DataFrame | Index):
        __logger__.debug("repr_object: → array")
        return repr_array(obj, **kwargs)
    if isinstance(x, Mapping):  # type: ignore[unreachable]
        __logger__.debug("repr_object: → mapping")
        return repr_mapping(obj, **kwargs)
    if isinstance(x, NTuple):
        __logger__.debug("repr_object: → namedtuple")
        return repr_namedtuple(obj, **kwargs)
    if isinstance(x, Sequence):
        __logger__.debug("repr_object: → sequence")
        return repr_sequence(obj, **kwargs)
    __logger__.debug("repr_object: → fallback %s", fallback)
    return fallback(obj)


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
    repr_fun: Callable[..., str] = NotImplemented,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a mapping object.

    Args:
        obj: Mapping object.
        align: Align keys and values.
        identifier: Identifier of the object.
        indent: Indentation level.
        linebreaks: Use linebreaks.
        maxitems: Maximum number of items to print.
        padding: Padding between keys and values.
        recursive: Recursively print values.
        repr_fun: Function to use for printing values.
        title: Title of the object.
        wrapped: Object to wrap the mapping in.

    Notes:
        - if recursive=True:  Name<Mapping>{key: repr_short(value), ...}
        - if recursive=False: Name<Mapping>{key: repr_object(value), ...}

        type only

             name<Mapping>

        value-type

            name<Mapping>:
                key1: repr_type(value1)
                key2: repr_type(value2)
                key3: repr_type(value3)

        fully recursive:

            name<Mapping>(
                key1: repr_func(value1)
                key2: repr_func(value2)
                key3: repr_func(value3)
    """
    # validate object
    if not isinstance(obj, Mapping):
        raise TypeError(f"Expected Mapping, got {type(obj)}.")

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

    # set separators
    br = "\n" if linebreaks else ""
    # key_sep = ": "
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    # set brackets
    left, right = "{", "}"

    # set max_key_length
    if align and linebreaks:
        max_key_length = max((len(str(key)) for key in obj), default=0)
    else:
        max_key_length = 0

    # set type
    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        if cls is dict:
            title = ""
        else:
            title = cls.__name__

    # set identifier
    if identifier is None:
        identifier = get_identifier(self) * bool(recursive)

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

    # create callable to stringify subitems
    if repr_fun is NotImplemented:
        repr_fun = cast(Callable[..., str], repr_object if recursive else repr_type)
    recurse = recursive if isinstance(recursive, bool) else recursive - 1 or False

    to_string = partial(
        repr_fun,
        align=align,
        indent=indent + padding,  # + max_key_length,
        padding=padding,
        recursive=recurse,
        # repr_fun=repr_fun,
        # wrapped=wrapped,
    )

    __logger__.debug(
        "repr_mapping: %s",
        {
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
            "wrapped": type(wrapped),
        },
    )

    # precompute the items
    head_half = maxitems // 2
    tail_half = max(len(obj) - head_half, head_half)
    head_items = [
        f"{str(key):<{max_key_length}}: {to_string(value)}"
        for i, (key, value) in enumerate(obj.items())
        if i < head_half
    ]
    tail_items = [
        f"{str(key):<{max_key_length}}: {to_string(value)}"
        for i, (key, value) in enumerate(obj.items())
        if i >= tail_half
    ]

    # assemble the string
    string = f"{title}{identifier}{left}{br}"
    if head_items:
        string += f"{sep}{br}".join(f"{pad}{item}" for item in head_items)
    if len(obj) > maxitems:
        string += f"{sep}{br}{pad}..."
    if tail_items:
        string += f"{sep}{br}"
        string += f"{sep}{br}".join(f"{pad}{item}" for item in tail_items)
    if linebreaks:  # trailing comma
        string += f"{sep}{br}"
    string += f"{br}{right}"

    # add indent
    string = string.replace("\n", "\n" + " " * indent)

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
    repr_fun: Callable[..., str] = NotImplemented,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a sequence object.

    - if recursive=`True`:  ``Name<Sequence>(repr_short(item1), ...)``
    - if recursive=`False`: ``Name<Sequence>(repr_object(item1), ...)``

    semi recursive::

        name<Sequence>(
            object<Sequence>(item1, item2, ...),
            object<tuple>(item1, item2, ...),
            object<Mapping>(key1: value1, ...),
        )

    fully recursive::

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
    # validate object
    if not isinstance(obj, Sequence):
        raise TypeError(f"Expected Sequence, got {type(obj)}.")

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

    # set separators
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    # set brackets
    if isinstance(obj, list):
        left, right = "[", "]"
    elif isinstance(obj, set):
        left, right = "{", "}"
    elif isinstance(obj, tuple):
        left, right = "(", ")"
    else:
        left, right = "[", "]"

    # set type
    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        if cls in (list, tuple, set):
            title = ""
        else:
            title = cls.__name__

    # set identifier
    if identifier is None:
        identifier = get_identifier(self) * bool(recursive)

    # create callable to stringify subitems
    if repr_fun is NotImplemented:
        repr_fun = cast(Callable[..., str], repr_object if recursive else repr_type)
    recurse = recursive if isinstance(recursive, bool) else recursive - 1 or False

    to_string = partial(
        repr_fun,
        align=align,
        indent=indent + padding,
        padding=padding,
        recursive=recurse,
        # repr_fun=repr_fun,
        # wrapped=None,
    )

    __logger__.debug(
        "repr_sequence: %s",
        {
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
            "wrapped": type(wrapped),
        },
    )

    # precompute the items
    # If len(obj) < maxitems//2, then tail_items is empty.
    head_half = maxitems // 2
    tail_half = max(len(obj) - head_half, head_half)
    head_items = [to_string(val) for val in obj[:head_half]]
    tail_items = [to_string(val) for val in obj[tail_half:]]

    # assemble the string
    string = f"{title}{identifier}{left}{br}"
    if head_items:
        string += f"{sep}{br}".join(f"{pad}{item}" for item in head_items)
    if len(obj) > maxitems:
        string += f"{sep}{br}{pad}..."
    if tail_items:
        string += f"{sep}{br}"
        string += f"{sep}{br}".join(f"{pad}{item}" for item in tail_items)
    if linebreaks:  # trailing comma
        string += f"{sep}{br}"
    string += f"{br}{right}"

    # add indent
    string = string.replace("\n", "\n" + " " * indent)

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
    repr_fun: Callable[..., str] = NotImplemented,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a dataclass object.

    - recursive=`False`:  ``Name<dataclass>(item1, item2, ...)``
    - recursive=`True`: ``Name<dataclass>(item1=repr(item1), item2=repr(item2), ...)``
    """
    if not is_dataclass(obj) or not isinstance(obj, Dataclass):
        raise TypeError(f"Expected Sequence, got {type(obj)}.")

    if repr_fun is NotImplemented:
        repr_fun = cast(Callable[..., str], repr_object if recursive else repr_type)

    fields = obj.__dataclass_fields__

    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        if self in BUILTIN_CONSTANTS:
            title = ""
        else:
            title = cls.__name__

    # set identifier
    if identifier is None:
        identifier = get_identifier(self) * bool(recursive)

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
            wrapped=wrapped,
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
        wrapped=wrapped,
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
    repr_fun: Callable[..., str] = NotImplemented,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    - recursive=True:  Name<tuple>(item1, item2, ...)
    - recursive=False: Name<tuple>(item1=repr(item1), item2=repr(item2), ...)
    """
    if not isinstance(obj, tuple) or not isinstance(obj, NTuple):
        raise TypeError(f"Expected NamedTuple, got {type(obj)}.")

    if repr_fun is NotImplemented:
        repr_fun = cast(Callable[..., str], repr_object if recursive else repr_type)

    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        if self in BUILTIN_CONSTANTS:
            title = ""
        else:
            title = cls.__name__

    # set identifier
    if identifier is None:
        identifier = get_identifier(self) * bool(recursive)

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
            wrapped=wrapped,
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
        wrapped=wrapped,
    )


def repr_type(
    obj: Any,
    /,
    *,
    identifier: Optional[str] = None,
    recursive: bool | int = False,
    **_: Any,
) -> str:
    r"""Return a string representation using an object's type."""
    __logger__.debug(
        "repr_type: %s, identifier=%s, recursive=%s, %s",
        type(obj),
        identifier,
        recursive,
        _,
    )

    if isinstance(obj, str):
        return obj
    if inspect.isclass(obj) or inspect.isbuiltin(obj):
        return repr(obj)
    for item in BUILTIN_CONSTANTS:
        if obj is item:
            return repr(obj)

    # set identifier
    if identifier is None:
        identifier = get_identifier(obj) * bool(recursive)

    is_type = isinstance(obj, type)
    obj_repr = obj.__name__ if is_type else obj.__class__.__name__
    return f"{obj_repr}{identifier}{()*(not is_type)}"


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

    if isinstance(obj, DataFrame | MultiIndex):
        dtypes = [repr_dtype(dtype) for dtype in obj.dtypes]
        string += ", " + repr_sequence(dtypes, linebreaks=False, maxitems=5)
    elif isinstance(obj, Index | Series | Array):
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


RECURSIVE_REPR_FUNS: list[ReprProtocol] = [
    repr_object,
    repr_mapping,
    repr_sequence,
    repr_dataclass,
    repr_namedtuple,
]
