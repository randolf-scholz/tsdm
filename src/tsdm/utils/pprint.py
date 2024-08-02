r"""Utility functions for string manipulation."""

__all__ = [
    # CONSTANTS
    "RECURSIVE_REPR_FUNS",
    "MAXITEMS",
    "MAXITEMS_INLINE",
    "LINEBREAKS",
    "INDENT",
    "RECURSIVE",
    "ALIGN",
    # ABCs & Protocols
    "Identifiers",
    "ReprProtocol",
    "Types",
    # Functions
    "get_identifier",
    "repr_array",
    "repr_dataclass",
    "repr_dtype",
    "repr_mapping",
    "repr_namedtuple",
    "repr_object",
    "repr_sequence",
    "repr_set",
    "repr_shortform",
]


import logging
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from dataclasses import is_dataclass
from enum import Enum
from math import prod
from types import FunctionType
from typing import Any, Final, Optional, Protocol

import numpy as np
import polars as pl
import pyarrow as pa
import torch
from pandas import ArrowDtype, DataFrame, MultiIndex
from pandas.core.dtypes.base import ExtensionDtype
from pyarrow import Array as PyArrowArray, Table as PyArrowTable

from tsdm.testing import (
    is_builtin,
    is_builtin_constant,
    is_builtin_type,
    is_na_value,
    is_scalar,
)
from tsdm.types.aliases import DType
from tsdm.types.dtypes import TYPESTRINGS
from tsdm.types.protocols import (
    Dataclass,
    NTuple,
    SupportsArray,
    SupportsDataframe,
    SupportsDevice,
    SupportsDtype,
    SupportsItem,
    SupportsShape,
)

__logger__: logging.Logger = logging.getLogger(__name__)

MAXITEMS: Final[int] = 20
r"""Default maxitems for repr_funcs."""
MAXITEMS_INLINE: Final[int] = 5
r"""Default maxitems for repr_funcs."""
LINEBREAKS: Final[bool] = True
r"""Default linebreaks for repr_funcs."""
INDENT: Final[int] = 4
r"""Default indentation for repr_funcs."""
RECURSIVE: Final[bool | int] = 1
r"""Default recursive for repr_funcs."""
ALIGN: Final[bool] = True
r"""Default align for repr_mapping."""


class Identifiers(Enum):
    r"""Enum for identifiers."""

    ARRAY = "<array>"
    DATACLASS = "<dataclass>"
    FUNCTION = "<function>"
    MAPPING = "<mapping>"
    NAMEDTUPLE = "<namedtuple>"
    SEQUENCE = "<sequence>"
    SET = "<set>"
    TABLE = "<table>"
    TYPE = "<type>"

    # Fallback
    NULL = ""


class Types(Enum):
    r"""Enum for types."""

    ARRAY = SupportsArray
    DATACLASS = Dataclass
    FUNCTION = FunctionType
    MAPPING = Mapping
    NAMEDTUPLE = NTuple
    SEQUENCE = Sequence
    SET = AbstractSet
    TABLE = SupportsDataframe
    TYPE = type


class ReprProtocol(Protocol):
    r"""Protocol for recursive repr functions."""

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
        padding: int = INDENT,
        recursive: bool | int = RECURSIVE,
        repr_fn: Callable[..., str] = NotImplemented,
        title: Optional[str] = None,
        wrapped: Optional[object] = None,
    ) -> str: ...


def get_identifier(obj: object, /, **_: Any) -> str:
    r"""Return the identifier of an object."""
    match obj:
        case type() as cls:
            return f"<{cls.__name__}>"
        case _ if is_builtin_type(type(obj)):
            return ""
        case SupportsArray():
            return "<array>"
        case SupportsDataframe():
            return "<table>"
        case Dataclass():
            return "<dataclass>"
        case NTuple():
            return "<namedtuple>"
        case Mapping():
            return "<mapping>"
        case Sequence():
            return "<sequence>"
        case FunctionType():
            return "<function>"
        case _:
            return ""


def repr_shortform(
    obj: object,
    /,
    *,
    identifier: Optional[str] = None,
    recursive: bool | int = False,
    **_: Any,
) -> str:
    r"""Return a shorthed string representation using an object's type."""
    match obj:
        case str(string):
            return string
        case type() as cls:
            return cls.__name__
        case builtin if is_builtin(builtin):
            return repr(builtin)
        case SupportsArray() as arr if arr.__array__().size <= 1:
            return repr(arr.__array__().item())
        case np.dtype() | torch.dtype() | ExtensionDtype() as dtype:
            return repr_dtype(dtype)
        case nan if is_na_value(nan):
            return repr(nan)
        case _:
            # set identifier
            if identifier is None:
                identifier = get_identifier(obj) * bool(recursive)

            return f"{obj.__class__.__name__}{identifier}()"


def repr_object(
    obj: object,
    /,
    *,
    fallback: Callable[..., str] = repr_shortform,
    **kwargs: Any,
) -> str:
    r"""Return a string representation of an object.

    Special casing for a bunch of cases.

    Examples:
        >>> repr_object(1)
        '1'
        >>> repr_object(3.14)
        '3.14'
    """
    x = kwargs.get("wrapped", obj)

    match x:
        case type() as cls:
            return cls.__name__
        case basic if is_scalar(basic):  # scalars, strings, etc.
            return repr(basic)
        case SupportsArray() as array:
            return repr_array(array, **kwargs)
        case Dataclass() as dtc:
            return repr_dataclass(dtc, **kwargs)
        case NTuple() as ntuple:
            return repr_namedtuple(ntuple, **kwargs)
        case Mapping() as mapping:
            return repr_mapping(mapping, **kwargs)
        case Sequence() as sequence:
            return repr_sequence(sequence, **kwargs)
        case AbstractSet() as set_:
            return repr_set(set_, **kwargs)
        case np.dtype() | torch.dtype() | ExtensionDtype() as dtype:
            return repr_dtype(dtype)
        case _:
            # avoid recursion:
            if fallback is repr:
                return object.__repr__(obj)
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
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] = NotImplemented,
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
        repr_fn: Function to use for printing values.
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

    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    # set title
    title = (
        str(title) if title is not None
        else cls.__name__
    )  # fmt: skip

    # set identifier
    identifier = (
        str(identifier) if identifier is not None
        else "<mapping>" if recursive
        else ""
    )  # fmt: skip

    # set linebreaks
    linebreaks = (
        False if len(obj) < 1  # always inline if empty mapping
        else bool(linebreaks) if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    # set maxitems
    maxitems = (
        int(maxitems) if maxitems is not None
        else MAXITEMS if linebreaks
        else MAXITEMS_INLINE
    )  # fmt: skip

    # set max_key_length
    max_key_length = (
        max((len(str(key)) for key in obj), default=0)
        if (align and linebreaks)
        else 0
    )  # fmt: skip

    # set separators
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks
    left, right = "(", ")"

    # special treatment for dict:
    if cls is dict:
        left, right = "{", "}"
        identifier = ""
        title = ""

    # set repr_func
    repr_fn = (
        repr_fn  # type: ignore[assignment]
        if repr_fn is not NotImplemented
        else repr_object
        if recursive
        else repr_shortform
    )

    # update recursive
    recursive = (
        recursive
        if isinstance(recursive, bool)
        else recursive - 1 or False
    )  # fmt: skip

    # construct the string-formatter
    def to_string(key: Any, value: Any, /) -> str:
        try:
            encoded_value = repr_fn(
                value,
                align=align,
                indent=indent + padding,
                padding=padding,
                recursive=recursive,
            )
        except Exception as exc:
            exc.add_note(
                f"repr_mapping[{key=!r}]: Failed to convert"
                f" value of type {type(value)!r} to string."
            )
            raise

        return f"{key!s:<{max_key_length}}: {encoded_value}"

    # precompute the item representations
    head_half = maxitems // 2
    tail_half = max(len(obj) - head_half, head_half)
    head_items = [
        to_string(key, value)
        for i, (key, value) in enumerate(obj.items())
        if i < head_half
    ]
    tail_items = [
        to_string(key, value)
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
    string += f"{right}"

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
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] = NotImplemented,
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

    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    # set title
    title = (
        str(title) if title is not None
        else cls.__name__
    )  # fmt: skip

    # set identifier
    identifier = (
        str(identifier) if identifier is not None
        else "<sequence>" if recursive
        else ""
    )  # fmt: skip

    # set linebreaks
    linebreaks = (
        False if len(obj) < 1  # always inline if empty mapping
        else bool(linebreaks) if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    # set maxitems
    maxitems = (
        int(maxitems) if maxitems is not None
        else MAXITEMS if linebreaks
        else MAXITEMS_INLINE
    )  # fmt: skip

    # set separators
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks
    left, right = "(", ")"

    # special case for list and tuple
    if cls is list:
        left, right = "[", "]"
        identifier = ""
        title = ""

    if cls is tuple:
        left, right = "(", ")"
        identifier = ""
        title = ""

    # set repr_fun
    repr_fn = (
        repr_fn  # type: ignore[assignment]
        if repr_fn is not NotImplemented
        else repr_object
        if recursive
        else repr_shortform
    )

    # update recursive
    recursive = (
        recursive
        if isinstance(recursive, bool)
        else recursive - 1 or False
    )  # fmt: skip

    # construct the string-formatter
    def to_string(index: int, value: Any, /) -> str:
        try:
            encoded_value = repr_fn(
                value,
                align=align,
                indent=indent + padding,
                padding=padding,
                recursive=recursive,
            )
        except Exception as exc:
            exc.add_note(
                f"repr_sequence[{index=!r}]: Failed to convert"
                f" value of type {type(value)!r} to string."
            )
            raise

        return encoded_value

    # precompute the item representations
    head_half = maxitems // 2
    tail_half = max(len(obj) - head_half, head_half)
    head_items = [to_string(i, v) for i, v in enumerate(obj[:head_half], start=0)]
    tail_items = [
        to_string(i, v) for i, v in enumerate(obj[tail_half:], start=tail_half)
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
    string += f"{right}"

    # add indent
    string = string.replace("\n", "\n" + " " * indent)

    return string


def repr_set(
    obj: AbstractSet,
    /,
    *,
    align: bool = ALIGN,
    identifier: Optional[str] = None,
    indent: int = 0,
    linebreaks: Optional[bool] = None,
    maxitems: Optional[int] = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] = NotImplemented,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a set-like object."""
    # validate object
    if not isinstance(obj, AbstractSet):
        raise TypeError(f"Expected Set, got {type(obj)}.")

    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    # set title
    title = (
        str(title) if title is not None
        else cls.__name__
    )  # fmt: skip

    # set identifier
    identifier = (
        str(identifier) if identifier is not None
        else "<set>" if recursive
        else ""
    )  # fmt: skip

    # set linebreaks
    linebreaks = (
        False if len(obj) < 1  # always inline if empty mapping
        else bool(linebreaks) if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    # set maxitems
    maxitems = (
        int(maxitems) if maxitems is not None
        else MAXITEMS if linebreaks
        else MAXITEMS_INLINE
    )  # fmt: skip

    # set separators
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks
    left, right = "(", ")"

    # special case set
    if cls is set:
        left, right = "{", "}"
        identifier = ""
        title = ""

    # special case frozenset
    if cls is frozenset:
        identifier = ""

    # set repr_fun
    repr_fn = (
        repr_fn  # type: ignore[assignment]
        if repr_fn is not NotImplemented
        else repr_object
        if recursive
        else repr_shortform
    )

    # update recursive
    recursive = (
        recursive
        if isinstance(recursive, bool)
        else recursive - 1 or False
    )  # fmt: skip

    # construct the string-formatter
    def to_string(index: int, value: Any, /) -> str:
        try:
            encoded_value = repr_fn(
                value,
                align=align,
                indent=indent + padding,
                padding=padding,
                recursive=recursive,
            )
        except Exception as exc:
            exc.add_note(
                f"repr_set[{index=!r}]: Failed to convert"
                f" value of type {type(value)!r} to string."
            )
            raise

        return encoded_value

    # precompute the item representations
    head_count = maxitems // 2
    tail_count = max(len(obj) - head_count, head_count)

    # setup iterators
    iterator = iter(obj)  # shared iterator
    head_iter = zip(range(head_count), iterator, strict=False)
    drop_iter = zip(range(head_count, tail_count - 1), iterator, strict=False)
    tail_iter = zip(range(tail_count, len(obj)), iterator, strict=False)
    head_items = [to_string(i, v) for i, v in head_iter]
    for _ in drop_iter:
        pass
    tail_items = [to_string(i, v) for i, v in tail_iter]

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
    string += f"{right}"

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
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] = NotImplemented,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a dataclass object.

    - recursive=`False`:  ``Name<dataclass>(item1, item2, ...)``
    - recursive=`True`: ``Name<dataclass>(item1=repr(item1), item2=repr(item2), ...)``
    """
    if not isinstance(obj, Dataclass):
        raise TypeError(f"Expected Dataclass, got {type(obj)}.")

    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    # set title
    title = (
        str(title) if title is not None
        else "" if is_builtin_constant(wrapped)
        else cls.__name__
    )  # fmt: skip

    # set identifier
    identifier = (
        str(identifier) if identifier is not None
        else "<dataclass>" if recursive
        else ""
    )  # fmt: skip

    # set repr_fun
    repr_fn = (
        repr_fn  # type: ignore[assignment]
        if repr_fn is not NotImplemented
        else repr_object
        if recursive
        else repr_shortform
    )

    fields = obj.__dataclass_fields__

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
            repr_fn=repr_fn,
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
        repr_fn=repr_fn,
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
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] = NotImplemented,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    - recursive=True:  Name<tuple>(item1, item2, ...)
    - recursive=False: Name<tuple>(item1=repr(item1), item2=repr(item2), ...)
    """
    if not isinstance(obj, NTuple):
        raise TypeError(f"Expected NamedTuple, got {type(obj)}.")

    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    # set title
    title = (
        str(title) if title is not None
        else "" if is_builtin_constant(wrapped)
        else cls.__name__
    )  # fmt: skip

    # set identifier
    identifier = (
        str(identifier) if identifier is not None
        else "<namedtuple>" if recursive
        else ""
    )  # fmt: skip

    # set repr_func
    repr_fn = (
        repr_fn  # type: ignore[assignment]
        if repr_fn is not NotImplemented
        else repr_object
        if recursive
        else repr_shortform
    )

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
            repr_fn=repr_fn,
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
        repr_fn=repr_fn,
        title=title,
        wrapped=wrapped,
    )


def repr_array(
    obj: SupportsArray,
    /,
    *,
    title: Optional[str] = None,
    maxitems: Optional[int] = MAXITEMS_INLINE,  # max number of dtypes to show
    **_: Any,
) -> str:
    r"""Return a string representation of an array object."""
    if not isinstance(obj, SupportsArray):
        raise TypeError("Object does not support `__array__` dunder.")

    maxitems = MAXITEMS_INLINE if maxitems is None else int(maxitems)

    # set type
    cls: type = obj.__class__

    # get the type-repr
    type_repr = str(title) if title is not None else cls.__name__

    # get the shape-repr
    # if multidimensional: <dim1, dim2, ...>
    # if plain scalar: none
    shape: tuple[int, ...] = (
        tuple(obj.shape)
        if isinstance(obj, SupportsShape)
        else tuple(obj.__array__().shape)
    )
    match len(shape):
        case 0:
            shape_repr = ""
        case _:
            shape_repr = f"<{','.join(str(dim) for dim in shape)}>"

    # get the device-repr
    match obj:
        case SupportsDevice(device=device):
            device_repr = f"@{device}"
        case _:
            device_repr = ""

    # get the dtype-repr
    # Table-like: [dtype1, dtype2, ...]
    # Tensor-like:
    match obj:
        # DataFrame-like
        case DataFrame(dtypes=dtypes) | MultiIndex(dtypes=dtypes):
            vals = [repr_dtype(dtype) for dtype in dtypes]
        case PyArrowTable() as table:
            vals = [repr_dtype(dtype) for dtype in table.schema.types]
        case pl.DataFrame(dtypes=dtypes):
            vals = [repr_dtype(dtype) for dtype in dtypes]
        case SupportsDataframe() as supports_frame:
            frame: DataFrame = supports_frame.__dataframe__()
            vals = [repr_dtype(dtype) for dtype in frame.dtypes]
        # Tensor-like
        case PyArrowArray(type=dtype):
            vals = [repr_dtype(dtype)]
        case SupportsDtype(dtype=dtype):
            vals = [repr_dtype(dtype)]
        case SupportsArray() as array:  # fallback
            vals = [repr_dtype(array.__array__().dtype)]
        case _:
            raise TypeError(f"Unsupported object type {type(obj)}.")

    # truncate the dtype-repr
    if len(vals) > maxitems:
        vals = vals[: maxitems // 2] + ["..."] + vals[-maxitems // 2 :]

    match vals, prod(shape):
        case _, 0 | 1:
            # skip for scalar
            dtype_repr = ""
        case _:
            dtype_repr = str(vals).replace("'", "")

    # determine the value-repr (show for scalars, otherwise hide)
    match prod(shape), obj:
        case 1, SupportsItem() as x:
            value_repr = f"({x.item()!s})"
        case 1, SupportsArray() as array:
            value_repr = f"({array.__array__().item()!s})"
        case 1, _:
            raise TypeError(f"Unsupported object type {type(obj)}.")
        case _:
            value_repr = ""
        # case _:
        #     raise TypeError(f"Unsupported object type {type(obj)}.")

    # Tensor<shape>dtype@device
    # Table<shape>[dtype1, dtype1, ...]@device
    # Scalar<shape>(value)@device
    # PlainScalar(value)@device
    return f"{type_repr}{shape_repr}{dtype_repr}{device_repr}{value_repr}"


def repr_dtype(
    dtype: str | type | DType | SupportsDtype | pa.DataType | pl.DataType, /
) -> str:
    r"""Return a string representation of a dtype object."""
    match dtype:
        case SupportsDtype(dtype=dtype):
            return repr_dtype(dtype)
        case str(string):
            return string
        # These are too verbose.
        case ArrowDtype() as wrapped_arrow_dtype:
            return repr_dtype(wrapped_arrow_dtype.pyarrow_dtype)
        # Some special casing for dictionary types.
        case pa.DictionaryType(index_type=index_type, value_type=value_type):
            return f"dict[{index_type!s},{value_type!s}]"  # type: ignore[has-type]
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
