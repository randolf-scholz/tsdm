r"""Utility functions for string manipulation."""

__all__ = [
    # CONSTANTS
    "RECURSIVE_REPR_FUNS",
    "MAXITEMS",
    "MAXITEMS_INLINE",
    "LINEBREAKS",
    "PADDING",
    "RECURSIVE",
    "ALIGN",
    # ABCs & Protocols
    "ReprProtocol",
    # Functions
    "get_identifier",
    "pprint_repr",
    "repr_array",
    "repr_dataclass",
    "repr_dtype",
    "repr_mapping",
    "repr_namedtuple",
    "repr_object",
    "repr_sequence",
    "repr_shortform",
]

import inspect
import logging
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from dataclasses import is_dataclass
from functools import partialmethod
from math import prod
from types import FunctionType

from pandas import DataFrame, MultiIndex
from pyarrow import Array as pyarrow_array, Table as pyarrow_table
from typing_extensions import Any, Final, Optional, Protocol, cast, overload

from tsdm.testing import is_builtin, is_builtin_constant, is_builtin_type, is_na_value
from tsdm.types.aliases import DType
from tsdm.types.dtypes import TYPESTRINGS
from tsdm.types.protocols import (
    Dataclass,
    NTuple,
    SupportsArray,
    SupportsDevice,
    SupportsDtype,
    SupportsItem,
    SupportsShape,
)
from tsdm.types.variables import T
from tsdm.utils.decorators import decorator

__logger__: logging.Logger = logging.getLogger(__name__)

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
    match obj:
        case type() as cls:
            return f"<{cls.__name__}>"
        case _ if is_builtin_type(type(obj)):
            # return f"<{cls.__name__}>"
            return ""
        case SupportsArray():
            return "<array>"
        case Dataclass():
            return "<dataclass>"
        case Mapping():
            return "<mapping>"
        case Sequence():
            return "<sequence>"
        case FunctionType():
            return "<function>"
        case _:
            return ""


def repr_shortform(
    obj: Any,
    /,
    *,
    identifier: Optional[str] = None,
    recursive: bool | int = False,
    **_: Any,
) -> str:
    r"""Return a shorthed string representation using an object's type."""
    __logger__.debug(
        "repr_type: %s, identifier=%s, recursive=%s, %s",
        type(obj),
        identifier,
        recursive,
        _,
    )

    match obj:
        case str() as string:
            return string
        case type() as cls:
            return cls.__name__
        case builtin if is_builtin(builtin):
            return repr(builtin)
        # case bool() | int() | float() | complex() | bytes() | slice() | range():
        #     return repr(obj)
        case SupportsArray() as arr if arr.__array__().size <= 1:
            return repr(arr.__array__().item())
        case nan if is_na_value(nan):
            return repr(nan)
        case _:
            # set identifier
            if identifier is None:
                identifier = get_identifier(obj) * bool(recursive)

            return f"{obj.__class__.__name__}{identifier}()"


def repr_object(
    obj: Any, /, *, fallback: Callable[..., str] = repr_shortform, **kwargs: Any
) -> str:
    r"""Return a string representation of an object.

    Special casing for a bunch of cases.
    """
    __logger__.debug("repr_object: %s, %s", type(obj), kwargs)

    x = kwargs.get("wrapped", obj)
    x = obj if x is None else obj

    match x:
        case str() as string:
            __logger__.debug("repr_object: → str")
            return string
        case builtin if inspect.isbuiltin(builtin):
            __logger__.debug("repr_object: → builtin")
            return repr(builtin)
        case type() as cls:
            __logger__.debug("repr_object: → type")
            return repr(cls)
        case SupportsArray() as array:
            __logger__.debug("repr_object: → array")
            return repr_array(array, **kwargs)
        case Dataclass() as dtc:
            __logger__.debug("repr_object: → dataclass")
            return repr_dataclass(dtc, **kwargs)
        case NTuple() as ntuple:
            __logger__.debug("repr_object: → namedtuple")
            return repr_namedtuple(ntuple, **kwargs)
        case Mapping() as mapping:
            __logger__.debug("repr_object: → mapping")
            return repr_mapping(mapping, **kwargs)
        case Sequence() as sequence:
            __logger__.debug("repr_object: → sequence")
            return repr_sequence(sequence, **kwargs)
        case _ if fallback is repr:
            __logger__.debug("repr_object: → fallback %s", fallback)
            return object.__repr__(obj)
        case _:
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
        maxitems = MAXITEMS if linebreaks else MAXITEMS_INLINE

    # set separators
    br = "\n" if linebreaks else ""
    # key_sep = ": "
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    # set brackets
    left, right = "{", "}"

    # set max_key_length

    max_key_length = (
        max((len(str(key)) for key in obj), default=0) if (align and linebreaks) else 0
    )

    # set type
    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        title = "" if cls is dict else cls.__name__

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
        repr_fun = cast(
            Callable[..., str], repr_object if recursive else repr_shortform
        )
    recurse = recursive if isinstance(recursive, bool) else recursive - 1 or False

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

    def to_string(key: Any, value: Any, /, *, justify: int = 0) -> str:
        """Encode key and value."""
        try:
            encoded_value = repr_fun(
                value,
                align=align,
                indent=indent + padding,
                padding=padding,
                recursive=recurse,
                # repr_fun=repr_fun,
                # wrapped=None,
            )
        except Exception as exc:
            exc.add_note(
                f"repr_mapping[{key=!r}]: Failed to convert"
                f" value of type {type(value)!r} to string."
            )
            raise

        return f"{key!s:<{justify}}: {encoded_value}"

    # precompute the items
    head_half = maxitems // 2
    tail_half = max(len(obj) - head_half, head_half)
    head_items = [
        to_string(key, value, justify=max_key_length)
        # f"{str(key):<{max_key_length}}: {to_string(value)}"
        for i, (key, value) in enumerate(obj.items())
        if i < head_half
    ]
    tail_items = [
        to_string(key, value, justify=max_key_length)
        # f"{str(key):<{max_key_length}}: {to_string(value)}"
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
        maxitems = MAXITEMS if linebreaks else MAXITEMS_INLINE

    # set separators
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks

    # set brackets
    match obj:
        case tuple():
            left, right = "(", ")"
        case AbstractSet():
            left, right = "{", "}"
        case _:
            left, right = "[", "]"

    # set type
    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        title = "" if cls in {list, tuple, set} else cls.__name__

    # set identifier
    if identifier is None:
        identifier = get_identifier(self) * bool(recursive)

    # create callable to stringify subitems
    if repr_fun is NotImplemented:
        repr_fun = cast(
            Callable[..., str], repr_object if recursive else repr_shortform
        )
    recurse = recursive if isinstance(recursive, bool) else recursive - 1 or False

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

    def to_string(index: int, value: Any, /) -> str:
        try:
            encoded_value = repr_fun(
                value,
                align=align,
                indent=indent + padding,
                padding=padding,
                recursive=recurse,
                # repr_fun=repr_fun,
                # wrapped=None,
            )
        except Exception as exc:
            exc.add_note(
                f"repr_sequence[{index}]: Failed to convert"
                f" value of type {type(value)!r} to string."
            )
            raise

        return encoded_value

    # precompute the items
    # If len(obj) < maxitems//2, then tail_items is empty.
    head_half = maxitems // 2
    tail_half = max(len(obj) - head_half, head_half)
    head_items = [to_string(*x) for x in enumerate(obj[:head_half], start=0)]
    tail_items = [to_string(*x) for x in enumerate(obj[tail_half:], start=tail_half)]

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
    obj: Dataclass,
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
        raise TypeError(f"Expected Dataclass, got {type(obj)}.")

    if repr_fun is NotImplemented:
        repr_fun = cast(
            Callable[..., str], repr_object if recursive else repr_shortform
        )

    fields = obj.__dataclass_fields__

    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        title = "" if is_builtin_constant(self) else cls.__name__

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
        repr_fun = cast(
            Callable[..., str], repr_object if recursive else repr_shortform
        )

    self = obj if wrapped is None else wrapped
    cls = type(self)

    # set title
    if title is None:
        title = "" if is_builtin_constant(self) else cls.__name__

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


def repr_array(
    obj: SupportsArray,
    /,
    *,
    title: Optional[str] = None,
    **_: Any,
) -> str:
    r"""Return a string representation of an array object."""
    if not isinstance(obj, SupportsArray):
        raise TypeError("Object does not support `__array__` dunder.")

    cls: type = obj.__class__
    title = cls.__name__ if title is None else title

    string = f"{title}["

    # add the shape.
    shape = obj.shape if isinstance(obj, SupportsShape) else obj.__array__().shape
    # if the object is scalar-like, try to get the actual value.
    if prod(shape) <= 1 and isinstance(obj, SupportsItem):
        string += repr(obj.item())
    else:
        string += str(tuple(shape))

    # add the dtype
    match obj:
        case DataFrame(dtypes=dtypes) | MultiIndex(dtypes=dtypes):
            dtypes = [repr_dtype(dtype) for dtype in dtypes]
            string += ", " + repr_sequence(dtypes, linebreaks=False, maxitems=5)
        case pyarrow_table() as table:
            dtypes = [repr_dtype(dtype) for dtype in table.schema.types]
            string += ", " + repr_sequence(dtypes, linebreaks=False, maxitems=5)
        case pyarrow_array(type=dtype):
            string += ", " + repr_dtype(dtype)
        case SupportsDtype(dtype=dtype):
            string += ", " + repr_dtype(dtype)
        case _:
            raise TypeError(f"Cannot get dtype of {type(obj)}")

    # add the device
    if isinstance(obj, SupportsDevice):
        # FIXME: mypy thinks it's Never
        string += f"@{obj.device}"  # type: ignore[attr-defined]

    string += "]"
    return string


def repr_dtype(dtype: str | type | DType | SupportsDtype, /) -> str:
    r"""Return a string representation of a dtype object."""
    match dtype:
        case SupportsDtype(dtype=dtype):
            return repr_dtype(dtype)
        case str() as string:
            return string
        case type() as cls if cls in TYPESTRINGS:
            return TYPESTRINGS[cls]
        case _:
            return str(dtype)


RECURSIVE_REPR_FUNS: list[ReprProtocol] = [
    repr_array,
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_object,
    repr_sequence,
    repr_shortform,
]


@overload
def pprint_repr(cls: type[T], /) -> type[T]: ...
@overload
def pprint_repr(**kwds: Any) -> Callable[[type[T]], type[T]]: ...
@decorator
def pprint_repr(cls, /, **kwds):
    """Add appropriate __repr__ to class."""
    assert isinstance(cls, type), "Must be a class!"
    repr_func: Callable[..., str]

    if is_dataclass(cls):
        repr_func = repr_dataclass
    elif issubclass(cls, NTuple):  # type: ignore[misc]
        repr_func = repr_namedtuple
    elif issubclass(cls, Mapping):
        repr_func = repr_mapping
    elif issubclass(cls, SupportsArray):
        repr_func = repr_array
    elif issubclass(cls, Sequence):
        repr_func = repr_sequence
    elif issubclass(cls, type):
        repr_func = repr_shortform
    else:
        raise TypeError(f"Unsupported type {cls}.")

    cls.__repr__ = partialmethod(repr_func, **kwds)  # pyright: ignore
    return cls
