r"""Pretty printing utility for generic types.

Provides functions for pretty printing

- Sequences
- Mappings
- Sets
- Dataclasses
- Namedtuples
- Arrays (numpy, torch, pandas, polars)
  (provides a short form representation that shows that shape and dtype, but not the data)
- DTypes (numpy, torch, pandas, polars)

The output format is of the form Type@Modifier<Identifier>(values)

Examples:
    Type<Mapping>(
        key1: value1,
        key2: value2,
        ...,
        keyN: valueN,
    )

    Type<Sequence>(
        value1,
        value2,
        ...,
        valueN,
    )

    Type<Set>(
        value1,
        value2,
        ...,
        valueN,
    )
"""

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
    "repr_generic",
    "repr_sequence",
    "repr_set",
    "repr_shortform",
    # decorators
    "pprint_dataclass",
    "pprint_mapping",
    "pprint_namedtuple",
    "pprint_repr",
    "pprint_sequence",
    "pprint_set",
]

import builtins
import dataclasses
import datetime as dt
import math
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from enum import Enum
from functools import partial
from inspect import isbuiltin as is_builtin_function
from math import prod
from types import FunctionType
from typing import Any, Final, Optional, Protocol, SupportsFloat

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch

from .datatools import get_dtypes
from .decorator import decorator
from .types import (
    Dataclass,
    NTuple,
    SupportsArray,
    SupportsDataFrame,
    SupportsDevice,
    SupportsDtype,
    SupportsItem,
    SupportsShape,
    is_dataclass,
    is_namedtuple,
)

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
    TABLE = SupportsDataFrame
    TYPE = type


class ReprProtocol(Protocol):
    r"""Protocol for recursive repr functions.

    The output string is of the form ``Title@Modifier<Identifier>[generics](values)``
    """

    def __call__(
        self,
        obj: Any,
        /,
        *,
        align: bool = ALIGN,
        modifier: Optional[str] = None,
        identifier: Optional[str] = None,
        indent: int = 0,
        linebreaks: Optional[bool] = None,
        maxitems: Optional[int] = None,
        padding: int = INDENT,
        recursive: bool | int = RECURSIVE,
        repr_fn: Optional[Callable[..., str]] = None,
        title: Optional[str] = None,
        wrapped: Optional[object] = None,
    ) -> str: ...


def _is_builtin_singleton(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    return (
        obj is None
        or obj is True
        or obj is False
        or obj is Ellipsis
        or obj is NotImplemented
    )


def _is_builtin(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    return (
        is_builtin_function(obj)
        or _is_builtin_singleton(obj)
        or (isinstance(obj, type) and obj in builtins.__dict__.values())
    )


def _is_scalar(obj: object, /) -> bool:
    r"""Check if an object is a basic type."""
    return (
        _is_builtin_singleton(obj)
        or isinstance(
            obj, bool | int | float | complex | str | bytes | dt.datetime | dt.timedelta
        )
        or np.isscalar(obj)
        or obj is pd.NA
        or obj is pd.NaT
    )


def _is_nan(obj: object, /) -> bool:
    r"""Check if an object is NaN."""
    return (
        math.isnan(float(obj))
        if isinstance(obj, SupportsFloat)
        else np.isnan(obj)
        if isinstance(obj, np.generic)
        else obj is pd.NA
    )


def _is_dtype(arg: object, /) -> bool:
    r"""Check if a string is a valid dtype."""
    return isinstance(
        arg,
        np.dtype
        | torch.dtype
        | pa.DataType
        | pd.api.extensions.ExtensionDtype
        | pl.DataType,
    ) or (isinstance(arg, type) and issubclass(arg, pl.DataType))


def get_identifier(obj: object, /, **_: Any) -> str:
    r"""Return the identifier of an object."""
    match obj:
        case type() as cls:
            return f"<{cls.__name__}>"
        case SupportsArray():
            return "<array>"
        case SupportsDataFrame():
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
        case builtin if _is_builtin(builtin):
            return repr(builtin)
        case SupportsArray() as arr if arr.__array__().size <= 1:
            return repr(arr.__array__().item())
        case dtype if _is_dtype(dtype):
            return repr_dtype(dtype)
        case nan if _is_nan(obj):
            return repr(nan)
        case _:
            # set identifier
            if identifier is None:
                identifier = get_identifier(obj) * bool(recursive)

            return f"{obj.__class__.__name__}{identifier}()"


def repr_generic(
    obj: object,
    /,
    *,
    fallback: Callable[..., str] = repr_shortform,
    **kwargs: Any,
) -> str:
    r"""Return a string representation of an object.

    Special casing for a bunch of cases.

    Examples:
        >>> repr_generic(1)
        '1'
        >>> repr_generic(3.14)
        '3.14'
    """
    x = kwargs.get("wrapped", obj)

    match x:
        case type() as cls:
            return cls.__name__
        case basic if _is_scalar(basic):  # scalars, strings, etc.
            return repr(basic)
        case pl.LazyFrame() as frame:
            return repr_array(frame, **kwargs)
        case SupportsArray() as array:
            return repr_array(array, **kwargs)
        case Dataclass() as dtc:  # pyrefly: ignore[unsafe-overlap]
            return repr_dataclass(dtc, **kwargs)
        case NTuple() as ntuple:  # pyrefly: ignore[unsafe-overlap]
            return repr_namedtuple(ntuple, **kwargs)
        case Mapping() as mapping:
            return repr_mapping(mapping, **kwargs)
        case Sequence() as sequence:
            return repr_sequence(sequence, **kwargs)
        case AbstractSet() as set_:
            return repr_set(set_, **kwargs)
        case dtype if _is_dtype(dtype):
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
    modifier: Optional[str] = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Optional[Callable[..., str]] = None,
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
        modifier: Modifier of the object.
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
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = title if title is not None else cls.__name__

    modifier = f"@{modifier}" if modifier is not None else ""

    identifier = (
        identifier if identifier is not None
        else "<mapping>" if recursive
        else ""
    )  # fmt: skip

    linebreaks = (
        False if len(obj) < 1  # always inline if empty mapping
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    maxitems = (
        maxitems if maxitems is not None
        else MAXITEMS if linebreaks
        else MAXITEMS_INLINE
    )  # fmt: skip

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
    repr_func = (
        repr_fn
        if repr_fn is not None
        else repr_generic
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
            encoded_value = repr_func(
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
    string = f"{title}{modifier}{identifier}{left}{br}"
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
    modifier: Optional[str] = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Optional[Callable[..., str]] = None,
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
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = title if title is not None else cls.__name__

    modifier = f"@{modifier}" if modifier is not None else ""

    identifier = (
        identifier if identifier is not None
        else "<sequence>" if recursive
        else ""
    )  # fmt: skip

    linebreaks = (
        False if len(obj) < 1  # always inline if empty mapping
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    maxitems = (
        maxitems if maxitems is not None
        else MAXITEMS if linebreaks
        else MAXITEMS_INLINE
    )  # fmt: skip

    # set separators
    br = "\n" if linebreaks else ""
    sep = "," if linebreaks else ", "
    pad = " " * padding * linebreaks
    left, right = "(", ")"

    # special case list and tuple
    if cls is list:
        left, right = "[", "]"
        identifier = ""
        title = ""

    if cls is tuple:
        left, right = "(", ")"
        identifier = ""
        title = ""

    # set repr_fun
    repr_func = (
        repr_fn
        if repr_fn is not None
        else repr_generic
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
            encoded_value = repr_func(
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
    string = f"{title}{modifier}{identifier}{left}{br}"
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
    modifier: Optional[str] = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Optional[Callable[..., str]] = None,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a set-like object."""
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = title if title is not None else cls.__name__

    modifier = f"@{modifier}" if modifier is not None else ""

    identifier = (
        identifier if identifier is not None
        else "<set>" if recursive
        else ""
    )  # fmt: skip

    linebreaks = (
        False if len(obj) < 1  # always inline if empty mapping
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    maxitems = (
        maxitems if maxitems is not None
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
    repr_func = (
        repr_fn
        if repr_fn is not None
        else repr_generic
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
            encoded_value = repr_func(
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
    string = f"{title}{modifier}{identifier}{left}{br}"

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
    modifier: Optional[str] = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Optional[Callable[..., str]] = None,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a dataclass object.

    - recursive=`False`:  ``Name<dataclass>(item1, item2, ...)``
    - recursive=`True`: ``Name<dataclass>(item1=repr(item1), item2=repr(item2), ...)``
    """
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = (
        title if title is not None
        else "" if _is_builtin_singleton(wrapped)
        else cls.__name__
    )  # fmt: skip

    identifier = (
        identifier if identifier is not None
        else "<dataclass>" if recursive
        else ""
    )  # fmt: skip

    fields = dataclasses.fields(obj)

    if recursive:
        return repr_mapping(
            {field.name: getattr(obj, field.name) for field in fields if field.repr},
            align=align,
            identifier=identifier,
            indent=indent,
            linebreaks=linebreaks,
            maxitems=maxitems,
            modifier=modifier,
            padding=padding,
            recursive=recursive,
            repr_fn=repr_fn,
            title=title,
            wrapped=wrapped,
        )

    return repr_sequence(
        [field.name for field in fields if field.repr],
        align=align,
        identifier=identifier,
        indent=indent,
        linebreaks=linebreaks,
        maxitems=maxitems,
        modifier=modifier,
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
    modifier: Optional[str] = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Optional[Callable[..., str]] = None,
    title: Optional[str] = None,
    wrapped: Optional[object] = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    - recursive=True:  Name<tuple>(item1, item2, ...)
    - recursive=False: Name<tuple>(item1=repr(item1), item2=repr(item2), ...)
    """
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = (
        title if title is not None
        else "" if _is_builtin_singleton(wrapped)
        else cls.__name__
    )  # fmt: skip

    identifier = (
        identifier if identifier is not None
        else "<namedtuple>" if recursive
        else ""
    )  # fmt: skip

    if recursive:
        return repr_mapping(
            obj._asdict(),
            align=align,
            identifier=identifier,
            indent=indent,
            linebreaks=linebreaks,
            maxitems=maxitems,
            modifier=modifier,
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
        modifier=modifier,
        padding=padding,
        recursive=recursive,
        repr_fn=repr_fn,
        title=title,
        wrapped=wrapped,
    )


def repr_array(
    obj: SupportsArray | pl.LazyFrame,
    /,
    *,
    modifier: Optional[str] = None,
    identifier: Optional[str] = None,
    title: Optional[str] = None,
    maxitems: Optional[int] = MAXITEMS_INLINE,  # max number of dtypes to show
    **_: Any,
) -> str:
    r"""Return a string representation of an array object."""
    maxitems = MAXITEMS_INLINE if maxitems is None else maxitems

    # set type
    cls: type = obj.__class__

    # get the type-repr
    type_repr = title if title is not None else cls.__name__

    if isinstance(obj, pl.LazyFrame):
        schema = obj.collect_schema()
        shape_repr = f"<*,{len(schema)}>" if identifier is None else identifier
        vals = [repr_dtype(dtype) for dtype in schema.values()]

        if len(set(vals)) <= 1:
            vals = vals[:1]
        if len(vals) > maxitems:
            vals = [*vals[: maxitems // 2], "...", *vals[-maxitems // 2 :]]

        dtype_repr = str(vals).replace("'", "")
        modifier_repr = f"@{modifier}" if modifier is not None else ""
        return f"{type_repr}{modifier_repr}{shape_repr}{dtype_repr}"

    # set identifier = shape
    # if multidimensional: <dim1, dim2, ...>
    # if plain scalar: none
    shape: tuple[int, ...] = (
        tuple(obj.shape)
        if isinstance(obj, SupportsShape)
        else tuple(obj.__array__().shape)
    )
    match identifier, len(shape):
        case None, _:
            shape_repr = "" if not shape else f"<{','.join(str(dim) for dim in shape)}>"
        case _:
            shape_repr = "" if identifier is None else identifier

    # set modifier = device
    match modifier, obj:
        case None, SupportsDevice(device=device):
            device_repr = f"@{device}"
        case _:
            device_repr = f"@{modifier}" if modifier is not None else ""

    # get the dtype-repr
    # Table-like: [dtype1, dtype2, ...]
    # Tensor-like:
    vals = [repr_dtype(dtype) for dtype in get_dtypes(obj)]

    # if all vals are the same, show only one
    if len(set(vals)) <= 1:
        vals = vals[:1]

    # truncate the dtype-repr
    if len(vals) > maxitems:
        vals = [*vals[: maxitems // 2], "...", *vals[-maxitems // 2 :]]

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

    # Tensor@device<shape>dtype
    # Table@device<shape>[dtype1, dtype1, ...]
    # Scalar@device<shape>(value)
    # PlainScalar@device(value)
    return f"{type_repr}{device_repr}{shape_repr}{dtype_repr}{value_repr}"


def repr_dtype(dtype: object, /) -> str:
    r"""Return a string representation of a dtype object."""
    match dtype:
        case SupportsDtype(dtype=dtype):
            return repr_dtype(dtype)
        case str(string):
            return string
        # These are too verbose.
        case pd.ArrowDtype() as wrapped_arrow_dtype:
            return repr_dtype(wrapped_arrow_dtype.pyarrow_dtype)
        # Some special casing for dictionary types.
        case pa.DictionaryType(index_type=index_type, value_type=value_type):
            return f"dict[{index_type!s},{value_type!s}]"
        case type() as cls:
            return cls.__name__
        case _:
            return str(dtype)


RECURSIVE_REPR_FUNS: list[ReprProtocol] = [
    repr_array,
    repr_dataclass,
    repr_mapping,
    repr_namedtuple,
    repr_generic,
    repr_sequence,
    repr_shortform,
]


@decorator  # pyrefly: ignore[bad-argument-type]
# def pprint_sequence[Seq: Sequence](cls: type[Seq], /, **kwds: Any) -> type[Seq]:
def pprint_sequence[Cls: type[Sequence]](cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Sequence):
        raise TypeError(f"Expected Sequence type, got {cls}.")

    cls.__repr__ = partial(repr_sequence, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
# def pprint_mapping[Map: Mapping](cls: type[Map], /, **kwds: Any) -> type[Map]:
def pprint_mapping[Cls: type[Mapping]](cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, Mapping):
        raise TypeError(f"Expected Mapping type, got {cls}.")

    cls.__repr__ = partial(repr_mapping, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
# def pprint_set[Set: AbstractSet](cls: type[Set], /, **kwds: Any) -> type[Set]:
def pprint_set[Cls: type[AbstractSet]](cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not issubclass(cls, AbstractSet):
        raise TypeError(f"Expected Set type, got {cls}.")

    cls.__repr__ = partial(repr_set, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
# def pprint_dataclass[T](cls: type[T], /, **kwds: Any) -> type[T]:
def pprint_dataclass[Cls: Dataclass](cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise TypeError(f"Expected Sequence type, got {cls}.")

    cls.__repr__ = partial(repr_dataclass, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
# def pprint_namedtuple[Ntp: NTuple](cls: type[Ntp], /, **kwds: Any) -> type[Ntp]:
def pprint_namedtuple[Cls: type[NTuple]](cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not (isinstance(cls, type) and is_namedtuple(cls)):
        raise TypeError(f"Expected NamedTuple type, got {cls}.")

    cls.__repr__ = partial(repr_namedtuple, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
# def pprint_repr[T](cls: type[T], /, **kwds: Any) -> type[T]:
def pprint_repr[Cls: type](cls: Cls, /, **kwds: Any) -> Cls:
    r"""Add appropriate __repr__ to class."""
    if not isinstance(cls, type):
        raise TypeError("Must be a class!")

    repr_func: Callable[..., str]

    if is_dataclass(cls):
        repr_func = repr_dataclass
    elif is_namedtuple(cls):
        repr_func = repr_namedtuple
    elif issubclass(cls, Mapping):
        repr_func = repr_mapping
    elif issubclass(cls, SupportsArray):
        repr_func = repr_array
    elif issubclass(cls, Sequence):
        repr_func = repr_sequence
    elif issubclass(cls, AbstractSet):
        repr_func = repr_set
    elif issubclass(cls, type):
        repr_func = repr_shortform
    else:
        raise TypeError(f"Unsupported type {cls}.")

    cls.__repr__ = partial(repr_func, **kwds)  # ty: ignore[invalid-assignment]
    return cls
