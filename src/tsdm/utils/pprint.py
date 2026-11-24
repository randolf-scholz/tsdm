r"""Pretty printing utility for generic types.

Provides functions for pretty printing:

- Sequences
- Mappings
- Sets
- Dataclasses
- Namedtuples
- Arrays (NumPy, PyTorch, pandas, Polars), summarized by shape and dtype;
    single-element arrays also show their value.
- DTypes (numpy, torch, pandas, polars)

The general output format is `Type@Modifier<Identifier>(values)`, with optional
components depending on the object and formatting options. Builtin containers
use their usual delimiters without a type prefix.

Examples:
    ```text
    Type<mapping>(
        key1: value1,
        key2: value2,
        ...,
        keyN: valueN,
    )

    Type<sequence>(
        value1,
        value2,
        ...,
        valueN,
    )

    Type<set>(
        value1,
        value2,
        ...,
        valueN,
    )
    ```
"""

__all__ = [
    # CONSTANTS
    "MAXITEMS",
    "MAXITEMS_INLINE",
    "LINEBREAKS",
    "INDENT",
    "RECURSIVE",
    "ALIGN",
    # ABCs & Protocols
    "Identifier",
    "ReprOptions",
    "ReprProtocol",
    # Functions
    "repr_object",
    "repr_compact",
    # decorators
    "pretty_repr",
]

import dataclasses
import datetime as dt
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet, Sized
from enum import StrEnum
from functools import partial
from itertools import islice
from math import prod
from types import BuiltinFunctionType, FunctionType
from typing import Any, Final, Protocol, Self, TypedDict, Unpack, cast

from tsdm.types import (
    DataclassType,
    HasDevice,
    HasDType,
    HasDTypes,
    HasShape,
    NamedTupleType,
    SupportsArray,
    SupportsDataFrame,
    SupportsItem,
    is_dataclass,
    is_namedtuple,
)

from .decorator import decorator

MAXITEMS: Final[int] = 20
r"""Default item limit for multiline representations."""
MAXITEMS_INLINE: Final[int] = 5
r"""Default item limit for inline representations."""
LINEBREAKS: Final[bool] = True
r"""Default preference for line breaks in representations."""
INDENT: Final[int] = 4
r"""Default number of spaces added at each nesting level."""
RECURSIVE: Final[bool | int] = 1
r"""Default recursion depth for nested representations."""
ALIGN: Final[bool] = True
r"""Whether mapping keys are aligned by default."""


class ReprOptions(TypedDict, total=False):
    r"""Keyword options shared by recursive representation functions."""

    align: bool
    identifier: str | None
    indent: int
    linebreaks: bool | None
    maxitems: int | None
    modifier: str | None
    padding: int
    recursive: bool | int
    repr_fn: Callable[..., str] | None
    title: str | None
    wrapped: object | None


class ReprProtocol(Protocol):
    r"""Protocol for recursive repr functions.

    The general output form is `Title@Modifier<Identifier>(values)`, with
    components and delimiters depending on the represented object.
    `ReprOptions` describes the keyword arguments as a dictionary.
    """

    def __call__(
        self,
        obj: Any,
        /,
        *,
        align: bool = ALIGN,
        modifier: str | None = None,
        identifier: str | None = None,
        indent: int = 0,
        linebreaks: bool | None = None,
        maxitems: int | None = None,
        padding: int = INDENT,
        recursive: bool | int = RECURSIVE,
        repr_fn: Callable[..., str] | None = None,
        title: str | None = None,
        wrapped: object | None = None,
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


def _repr_python_scalar(arg: object, /) -> str | None:
    r"""Format known scalars, using native repr for datetime and timedelta subclasses."""
    if _is_builtin_singleton(arg):
        return repr(arg)

    cls = type(arg)
    if issubclass(cls, str):
        return str.__repr__(cast("str", arg))
    if issubclass(cls, bytes):
        return bytes.__repr__(cast("bytes", arg))
    if issubclass(cls, int):
        return int.__repr__(cast("int", arg))
    if issubclass(cls, float):
        return float.__repr__(cast("float", arg))
    if issubclass(cls, complex):
        return complex.__repr__(cast("complex", arg))
    if issubclass(cls, (dt.datetime, dt.timedelta)):
        # Subclasses can carry state the builtin repr loses, such as nanoseconds.
        return repr(arg)
    return None


def _repr_number(arg: object, /) -> str | None:
    r"""Format a Python number, preserving the type of zero-dimensional scalars."""
    # Extract the stored scalar first. Numeric hooks can turn booleans into
    # floats or discard the imaginary part of complex values.
    try:
        item_repr = _repr_python_scalar(arg.item()) if arg.shape == () else None  # type: ignore
    except Exception:
        item_repr = None
    if item_repr is not None:
        return item_repr

    # convert scalar-likes to python scalars
    try:
        return int.__repr__(arg.__index__())  # type: ignore
    except Exception:  # ruff: ignore[S110]
        pass
    try:
        return float.__repr__(arg.__float__())  # type: ignore
    except Exception:  # ruff: ignore[S110]
        pass
    try:
        return complex.__repr__(arg.__complex__())  # type: ignore
    except Exception:  # ruff: ignore[S110]
        pass
    return None


def _repr_dtype(dtype: object, /) -> str:
    r"""Return a string representation of a dtype object."""
    match dtype:
        case str(string):
            # strip leading module e.g. `torch.int32` -> `int32`
            string = string.removeprefix("torch.")
            # strip trailing `[pyarrow]`, e.g. `int32[pyarrow]` -> `int32`
            string = string.removesuffix("[pyarrow]")
            return string

        case object(dtype=dtype):  # type: ignore
            return _repr_dtype(dtype)

        case object(index_type=index_type, value_type=value_type):  # type: ignore
            # special casing for pa.DictionaryType
            return f"dict[{index_type!s},{value_type!s}]"

        case type() as cls:
            return cls.__name__

        case _:
            return _repr_dtype(str(dtype))


def _repr_identifier(label: str, /) -> str:
    r"""Enclose a nonempty label in angle brackets."""
    return f"<{label}>" if label else ""


def _item_bounds(size: int, maxitems: int, /) -> tuple[int, int]:
    r"""Return the exclusive head end and inclusive tail start."""
    if maxitems < 0:
        raise ValueError("maxitems must be nonnegative.")
    if size <= maxitems:
        return size, size
    return (maxitems + 1) // 2, size - maxitems // 2


def _join_items(
    head: Sequence[str],
    tail: Sequence[str],
    /,
    *,
    truncated: bool,
    linebreaks: bool = False,
    padding: int = INDENT,
) -> str:
    r"""Join rendered items, inserting an ellipsis for the omitted middle."""
    items = [*head, "...", *tail] if truncated else head
    if not linebreaks:
        return ", ".join(items)
    pad = " " * padding
    return "".join(f"\n{pad}{item}," for item in items) + "\n"


def _child_formatter(
    repr_fn: Callable[..., str] | None,
    /,
    *,
    recursive: bool | int,
    linebreaks: bool | None,
    maxitems: int | None,
) -> Callable[..., str]:
    r"""Select the formatter for children, forwarding unresolved layout options."""
    if repr_fn is not None:
        return repr_fn
    if recursive:
        return partial(repr_object, linebreaks=linebreaks, maxitems=maxitems)
    return repr_compact


def _repr_mapping(
    obj: Mapping,
    /,
    *,
    align: bool = ALIGN,
    identifier: str | None = None,
    indent: int = 0,
    linebreaks: bool | None = None,
    maxitems: int | None = None,
    modifier: str | None = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] | None = None,
    title: str | None = None,
    wrapped: object | None = None,
) -> str:
    r"""Return a string representation of a mapping object.

    Args:
        obj: Mapping object.
        align: Align keys and values.
        identifier: Custom identifier displayed in angle brackets.
        indent: Number of spaces prefixed to continuation lines.
        linebreaks: Whether to split items across lines. If omitted, follow
            `recursive`. Empty mappings are always inline.
        maxitems: Maximum number of entries to display, excluding the ellipsis.
        modifier: Modifier of the object.
        padding: Number of spaces added at each nesting level.
        recursive: Whether to format nested values recursively, or an integer
            limiting the recursion depth.
        repr_fn: Function to use for printing values.
        title: Title of the object.
        wrapped: Object whose type supplies the default title and delimiters.

    Note:
        Unless `repr_fn` is supplied, recursive formatting uses `repr_generic`
        for values; nonrecursive formatting uses `repr_shortform`. Keys are
        rendered with `str`. Builtin dictionaries use braces and only display
        explicitly supplied identifiers.

    Examples:
        Schematic output for a custom mapping:

        ```text
        Name<mapping>(
            key1: value1,
            key2: value2,
        )
        ```
    """
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = title if title is not None else cls.__name__

    modifier = f"@{modifier}" if modifier is not None else ""

    size = len(obj)

    multiline = (
        False if size == 0  # always inline if empty mapping
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    item_limit = (
        maxitems if maxitems is not None
        else MAXITEMS if multiline
        else MAXITEMS_INLINE
    )  # fmt: skip

    max_key_length = (
        max((len(str(key)) for key in obj), default=0)
        if (align and multiline)
        else 0
    )  # fmt: skip

    left, right = "(", ")"

    # special treatment for dict:
    if cls is dict:
        left, right = "{", "}"
        title = ""
    elif identifier is None and recursive:
        identifier = Identifier.get(wrapped)

    # set repr_func
    repr_func = _child_formatter(
        repr_fn, recursive=recursive, linebreaks=linebreaks, maxitems=maxitems
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

    # compute the item representations
    head_end, tail_start = _item_bounds(size, item_limit)
    items = iter(obj.items())
    head = [to_string(key, value) for key, value in islice(items, head_end)]
    tail = (
        [to_string(k, value) for k, value in islice(items, tail_start - head_end, None)]
        if tail_start < size
        else []
    )
    body = _join_items(
        head,
        tail,
        truncated=head_end < tail_start,
        linebreaks=multiline,
        padding=padding,
    )

    # assemble the string and add indentation
    label = _repr_identifier(identifier or "")
    string = f"{title}{modifier}{label}{left}{body}{right}"
    string = string.replace("\n", "\n" + " " * indent)

    return string


def _repr_sequence(
    obj: Sequence,
    /,
    *,
    align: bool = ALIGN,
    identifier: str | None = None,
    indent: int = 0,
    linebreaks: bool | None = None,
    maxitems: int | None = None,
    modifier: str | None = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] | None = None,
    title: str | None = None,
    wrapped: object | None = None,
) -> str:
    r"""Return a string representation of a sequence object.

    Unless `repr_fn` is supplied, `recursive=False` formats items with
    `repr_shortform`, while `recursive=True` uses `repr_generic`. A positive
    integer limits the recursion depth. With `linebreaks=None`, recursive
    representations use line breaks; empty sequences are always inline.

    Builtin lists use brackets and tuples use parentheses, without a title or
    identifier. Custom sequences use their type name and an optional
    `<sequence>` identifier. Long sequences show head and tail items separated
    by an ellipsis.

    Examples:
        Schematic nonrecursive output abbreviates nested containers:

        ```text
        Name<sequence>(list(), dict())
        ```

        Recursive output expands nested values, with line breaks when enabled:

        ```text
        Name<sequence>(
            [
                1,
                2,
            ],
            {
                key: value,
            },
        )
        ```
    """
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = title if title is not None else cls.__name__

    modifier = f"@{modifier}" if modifier is not None else ""

    size = len(obj)

    multiline = (
        False if size == 0  # always inline if empty sequence
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    item_limit = (
        maxitems if maxitems is not None
        else MAXITEMS if multiline
        else MAXITEMS_INLINE
    )  # fmt: skip

    left, right = "(", ")"

    # special case list and tuple
    if cls is list:
        left, right = "[", "]"
        title = ""
    elif cls is tuple:
        title = ""
    elif identifier is None and recursive:
        identifier = Identifier.get(wrapped)

    # set repr_fun
    repr_func = _child_formatter(
        repr_fn, recursive=recursive, linebreaks=linebreaks, maxitems=maxitems
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

    # compute the item representations
    head_end, tail_start = _item_bounds(size, item_limit)
    head = [to_string(i, value) for i, value in enumerate(obj[:head_end])]
    tail = [to_string(i, value) for i, value in enumerate(obj[tail_start:], tail_start)]
    body = _join_items(
        head,
        tail,
        truncated=head_end < tail_start,
        linebreaks=multiline,
        padding=padding,
    )

    # assemble the string and add indentation
    label = _repr_identifier(identifier or "")
    string = f"{title}{modifier}{label}{left}{body}{right}"
    string = string.replace("\n", "\n" + " " * indent)

    return string


def _repr_set(
    obj: AbstractSet,
    /,
    *,
    align: bool = ALIGN,
    identifier: str | None = None,
    indent: int = 0,
    linebreaks: bool | None = None,
    maxitems: int | None = None,
    modifier: str | None = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] | None = None,
    title: str | None = None,
    wrapped: object | None = None,
) -> str:
    r"""Return a string representation of a set-like object."""
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = title if title is not None else cls.__name__

    modifier = f"@{modifier}" if modifier is not None else ""

    size = len(obj)

    multiline = (
        False if size == 0  # always inline if empty set
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    item_limit = (
        maxitems if maxitems is not None
        else MAXITEMS if multiline
        else MAXITEMS_INLINE
    )  # fmt: skip

    left, right = "(", ")"

    # special case set
    if cls is set:
        left, right = "{", "}"
        title = ""
    elif cls is not frozenset and identifier is None and recursive:
        identifier = Identifier.get(wrapped)

    # set repr_fun
    repr_func = _child_formatter(
        repr_fn, recursive=recursive, linebreaks=linebreaks, maxitems=maxitems
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

    # compute the item representations
    head_end, tail_start = _item_bounds(size, item_limit)
    items = enumerate(obj)
    head = [to_string(i, value) for i, value in islice(items, head_end)]
    tail = (
        [to_string(i, value) for i, value in islice(items, tail_start - head_end, None)]
        if tail_start < size
        else []
    )
    body = _join_items(
        head,
        tail,
        truncated=head_end < tail_start,
        linebreaks=multiline,
        padding=padding,
    )

    # assemble the string and add indentation
    label = _repr_identifier(identifier or "")
    string = f"{title}{modifier}{label}{left}{body}{right}"
    string = string.replace("\n", "\n" + " " * indent)

    return string


def _repr_dataclass(
    obj: DataclassType,
    /,
    *,
    align: bool = ALIGN,
    identifier: str | None = None,
    indent: int = 0,
    linebreaks: bool | None = None,
    maxitems: int | None = None,
    modifier: str | None = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] | None = None,
    title: str | None = None,
    wrapped: object | None = None,
) -> str:
    r"""Return a string representation of a dataclass object.

    With `recursive=False`, show field names without values. Otherwise, format
    a mapping from field names to values recursively. Fields marked
    `repr=False` are omitted in both cases.
    """
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = (
        title if title is not None
        else "" if _is_builtin_singleton(wrapped)
        else cls.__name__
    )  # fmt: skip

    if identifier is None:
        identifier = (Identifier.get(wrapped) if recursive else None) or ""

    fields = dataclasses.fields(obj)

    if recursive:
        return _repr_mapping(
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

    return _repr_sequence(
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


def _repr_namedtuple(
    obj: NamedTupleType,
    /,
    *,
    align: bool = ALIGN,
    identifier: str | None = None,
    indent: int = 0,
    linebreaks: bool | None = None,
    maxitems: int | None = None,
    modifier: str | None = None,
    padding: int = INDENT,
    recursive: bool | int = RECURSIVE,
    repr_fn: Callable[..., str] | None = None,
    title: str | None = None,
    wrapped: object | None = None,
) -> str:
    r"""Return a string representation of a namedtuple object.

    With `recursive=False`, show field names without values. Otherwise, format
    the named tuple's field-value mapping recursively. Recursive output uses the
    `<namedtuple>` identifier by default.
    """
    # set type & wrapped
    wrapped = obj if wrapped is None else wrapped
    cls = type(wrapped)

    title = (
        title if title is not None
        else "" if _is_builtin_singleton(wrapped)
        else cls.__name__
    )  # fmt: skip

    if identifier is None:
        identifier = (Identifier.get(wrapped) if recursive else None) or ""

    if recursive:
        return _repr_mapping(
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

    return _repr_sequence(
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


def _repr_table(
    obj: SupportsDataFrame | HasDTypes,
    /,
    *,
    modifier: str | None = None,
    identifier: str | None = None,
    title: str | None = None,
    maxitems: int | None = MAXITEMS_INLINE,
    show_device: bool = False,
    **_: Any,
) -> str:
    r"""Summarize a table's shape and column dtypes without formatting its cells.

    Supports pandas and Polars dataframes, Polars lazy frames, PyArrow tables,
    and objects that provide the `__dataframe__` interchange protocol.
    Column dtypes determine the width. The row count comes from `.shape` or
    `len()` when available, and displays as `*` otherwise.

    Examples:
        ```pycon
        >>> from types import SimpleNamespace
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> data = {"name": ["foo", "bar"], "value": [0.2, 0.3], "ID": [1, 2]}
        >>> _repr_table(pd.DataFrame(data))
        'DataFrame<2,3>[str, float64, int64]'
        >>> _repr_table(pl.DataFrame(data))
        'DataFrame<2,3>[String, Float64, Int64]'
        >>> _repr_table(pl.LazyFrame(data))
        'LazyFrame<*,3>[String, Float64, Int64]'
        >>> _repr_table(pa.table(data))
        'Table<2,3>[STRING8, FLOAT64, INT64]'
        >>> view = SimpleNamespace(__dataframe__=pa.table(data).__dataframe__)
        >>> _repr_table(view)
        'SimpleNamespace<*,3>[STRING8, FLOAT64, INT64]'

        ```
    """
    match obj:  # determine dtypes
        case object(dtypes=native_dtypes):  # type: ignore
            dtypes = list(native_dtypes)
        case SupportsDataFrame():
            frame = obj.__dataframe__()
            # A protocol dtype is (kind, bit width, format string, endianness).
            protocol_dtypes = (column.dtype for column in frame.get_columns())
            dtypes = [f"{kind.name}{width}" for kind, width, _, _ in protocol_dtypes]
        case _:
            raise TypeError

    num_cols = len(dtypes)
    match obj:  # determine shape
        case object(shape=tuple(shape)):  # type: ignore
            num_rows, width = shape
            assert width == num_cols
        case Sized():
            num_rows = len(obj)
        case _:
            num_rows = None

    match modifier, show_device, obj:
        case None, True, HasDevice(device=device):
            modifier = str(device)

    type_repr = type(obj).__name__ if title is None else title
    modifier_repr = f"@{modifier}" if modifier is not None else ""
    rows_repr = "*" if num_rows is None else f"{num_rows}"
    shape_repr = (
        f"<{rows_repr},{num_cols}>"
        if identifier is None
        else _repr_identifier(identifier)
    )

    maxitems = MAXITEMS_INLINE if maxitems is None else maxitems
    labels = [_repr_dtype(dtype) for dtype in dtypes]
    if len(set(labels)) <= 1:
        labels = labels[:1]
    head_end, tail_start = _item_bounds(len(labels), maxitems)
    dtypes_repr = _join_items(
        labels[:head_end], labels[tail_start:], truncated=head_end < tail_start
    )

    return f"{type_repr}{modifier_repr}{shape_repr}[{dtypes_repr}]"


def _repr_array(
    obj: SupportsArray,
    /,
    *,
    modifier: str | None = None,
    identifier: str | None = None,
    title: str | None = None,
    show_device: bool = False,
    **_: Any,
) -> str:
    r"""Summarize an array by shape and dtype, showing its only value if available."""
    match obj:  # determine the shape
        case object(shape=tuple(shape)):  # type: ignore
            array = None
        case _:
            array = obj.__array__()
            shape = array.shape

    match obj:  # determine the device
        case object(device=device):  # type: ignore
            pass
        case _:
            device = None

    match obj:  # determine the dtype
        case object(dtype=dtype) | object(type=dtype):  # type: ignore
            # PyArrow arrays expose their native dtype as `.type`.
            pass
        case _:
            array = obj.__array__() if array is None else array
            dtype = array.dtype

    type_repr = type(obj).__name__ if title is None else title
    device_repr = (
        "" if not show_device else f"@{device}" if modifier is None else f"@{modifier}"
    )
    dtype_repr = _repr_dtype(dtype)
    shape_repr = (
        _repr_identifier(identifier)
        if identifier is not None
        else f"<{','.join(str(dim) for dim in shape)}>"
        if shape
        else ""
    )

    match prod(shape):
        case 0:
            suffix = ""
        case 1:
            source = (
                obj
                if isinstance(obj, SupportsItem)
                else obj.__array__()
                if array is None
                else array
            )
            try:
                value = source.item()
            except Exception:
                # Some arrays expose metadata even when their contents are unavailable.
                suffix = f"[{dtype_repr}]"
            else:
                suffix = f"({value!s})"
        case _:
            suffix = f"[{dtype_repr}]"

    return f"{type_repr}{device_repr}{shape_repr}{suffix}"


def _repr_type(obj: type, /, *, identifier: str | None = None, **_: Any) -> str:
    r"""Return a compact representation of a type."""
    label = _repr_identifier(identifier or "")
    return f"{obj.__name__}{label}"


def _repr_function(
    obj: FunctionType | BuiltinFunctionType,
    /,
    *,
    identifier: str | None = None,
    **_: Any,
) -> str:
    r"""Format a Python or built-in function with any supplied identifier."""
    label = _repr_identifier(identifier or "")
    if type(obj) is BuiltinFunctionType:
        return f"{label}{obj!r}"
    return f"{obj.__name__}{label}"


def repr_compact(
    obj: object,
    /,
    **options: Unpack[ReprOptions],
) -> str:
    r"""Return a compact representation using a value or its type name."""
    identifier = options.get("identifier")
    obj_type = type(obj)
    for entry in (Identifier.TYPE, Identifier.FUNCTION):
        if issubclass(obj_type, entry.kind):
            return entry.formatter(obj, identifier=identifier)

    label = _repr_identifier(identifier or "")
    if (scalar_repr := _repr_python_scalar(obj)) is not None:
        return f"{label}{scalar_repr}"
    if (number_repr := _repr_number(obj)) is not None:
        return f"{label}{number_repr}"

    wrapped = options.get("wrapped")
    displayed = obj if wrapped is None else wrapped
    if identifier is None and options.get("recursive", False):
        label = _repr_identifier(Identifier.get(displayed) or "")
    return f"{type(displayed).__name__}{label}()"


def repr_object(
    obj: object,
    /,
    *,
    fallback: Callable[..., str] = repr_compact,
    **options: Unpack[ReprOptions],
) -> str:
    r"""Return a string representation of an object.

    Dispatch to specialized formatters for scalars, arrays, containers,
    dataclasses, named tuples, and dtypes. Use `fallback` for unsupported objects.
    With `identifier=None`, each formatter chooses its automatic label.
    An explicit string is displayed as given inside angle brackets, and an
    empty string suppresses the label. Container formatters use the type of
    `wrapped` for automatic labels when a wrapper is supplied.

    For containers and table dtype summaries, `maxitems` is a nonnegative
    limit on displayed entries, excluding the ellipsis. When entries are
    omitted, an odd limit gives the head one more entry than the tail.
    Built-in recursive formatting forwards explicit `linebreaks` and
    `maxitems` values to each nested formatter. A value of `None` lets each
    formatter choose its default.

    Examples:
        ```pycon
        >>> repr_object(1)
        '1'
        >>> repr_object(3.14)
        '3.14'

        ```
    """
    label = _repr_identifier(options.get("identifier") or "")

    if (scalar_repr := _repr_python_scalar(obj)) is not None:
        return f"{label}{scalar_repr}"
    if (entry := Identifier.get(obj)) is not None:
        return entry.formatter(obj, **options)

    if fallback is repr_compact:
        return repr_compact(obj, **options)
    # Avoid calling an unsupported object's repr method again.
    if fallback is repr:
        return f"{label}{object.__repr__(obj)}"
    return f"{label}{fallback(obj)}"


class Identifier(StrEnum):
    r"""Labels, runtime types, and formatters in dispatch order.

    Look up a member by label with `Identifier("array")`. Use
    `Identifier.get(obj)` to select a member for an object.
    """

    kind: type | tuple[type, ...]
    formatter: ReprProtocol

    TYPE = "type", type, _repr_type
    FUNCTION = "function", (FunctionType, BuiltinFunctionType), _repr_function
    TABLE = "table", SupportsDataFrame, _repr_table
    ARRAY = "array", SupportsArray, _repr_array
    DATACLASS = "dataclass", DataclassType, _repr_dataclass
    NAMEDTUPLE = "namedtuple", NamedTupleType, _repr_namedtuple
    MAPPING = "mapping", Mapping, _repr_mapping
    SEQUENCE = "sequence", Sequence, _repr_sequence
    SET = "set", AbstractSet, _repr_set

    def __new__(
        cls, label: str, kind: type | tuple[type, ...], formatter: ReprProtocol
    ) -> Self:
        entry = str.__new__(cls, label)
        entry._value_ = label
        entry.kind = kind
        entry.formatter = formatter
        return entry

    @classmethod
    def get(cls, arg: object, /) -> Identifier | None:
        r"""Select the first applicable category."""
        for entry in cls:
            match entry:
                case Identifier.SEQUENCE if isinstance(arg, str):
                    continue
                case _ if isinstance(arg, entry.kind):
                    return entry
                # Tables can expose native column dtypes without interchange.
                case Identifier.TABLE if (
                    isinstance(arg, HasDTypes)
                    and not isinstance(arg, HasDType)
                    and (not isinstance(arg, HasShape) or len(arg.shape) == 2)
                ):
                    return entry
        return None


@decorator  # pyrefly: ignore[bad-argument-type]
def _pretty_sequence[Cls: type[Sequence]](
    cls: Cls, /, **kwds: Unpack[ReprOptions]
) -> Cls:
    r"""Install a `__repr__` using `repr_sequence` on a Sequence subclass."""
    if not issubclass(cls, Sequence):
        raise TypeError(f"Expected Sequence type, got {cls}.")

    cls.__repr__ = partial(_repr_sequence, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
def _pretty_mapping[Cls: type[Mapping]](
    cls: Cls, /, **kwds: Unpack[ReprOptions]
) -> Cls:
    r"""Install a `__repr__` using `repr_mapping` on a Mapping subclass."""
    if not issubclass(cls, Mapping):
        raise TypeError(f"Expected Mapping type, got {cls}.")

    cls.__repr__ = partial(_repr_mapping, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
def _pretty_set[Cls: type[AbstractSet]](
    cls: Cls, /, **kwds: Unpack[ReprOptions]
) -> Cls:
    r"""Install a `__repr__` using `repr_set` on a Set subclass."""
    if not issubclass(cls, AbstractSet):
        raise TypeError(f"Expected Set type, got {cls}.")

    cls.__repr__ = partial(_repr_set, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
def _pretty_dataclass[Cls: DataclassType](
    cls: Cls, /, **kwds: Unpack[ReprOptions]
) -> Cls:
    r"""Install a `__repr__` using `repr_dataclass` on a dataclass."""
    if not (isinstance(cls, type) and is_dataclass(cls)):
        raise TypeError(f"Expected Sequence type, got {cls}.")

    cls.__repr__ = partial(_repr_dataclass, **kwds)
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
def _pretty_namedtuple[Cls: type[NamedTupleType]](
    cls: Cls, /, **kwds: Unpack[ReprOptions]
) -> Cls:
    r"""Install a `__repr__` using `repr_namedtuple` on a named tuple class."""
    if not (isinstance(cls, type) and is_namedtuple(cls)):
        raise TypeError(f"Expected NamedTuple type, got {cls}.")

    cls.__repr__ = partial(_repr_namedtuple, **kwds)  # ty: ignore[invalid-assignment]
    return cls


@decorator  # pyrefly: ignore[bad-argument-type]
def pretty_repr[Cls: type](cls: Cls, /, **kwds: Unpack[ReprOptions]) -> Cls:
    r"""Install a `__repr__` that formats instances through `repr_object`.

    Forward `kwds` to `repr_object`, which selects a formatter in `Identifier`
    order based on the instance. Return the modified class, or raise
    `TypeError` for unsupported classes and nonclass inputs.
    """
    if not isinstance(cls, type):
        raise TypeError("Must be a class!")

    # Native table metadata can be exposed by a class attribute or descriptor.
    if not (
        isinstance(cls, HasDTypes)
        or any(issubclass(cls, entry.kind) for entry in Identifier)
    ):
        raise TypeError(f"Unsupported type {cls}.")

    cls.__repr__ = partial(repr_object, **kwds)  # ty: ignore[invalid-assignment]
    return cls
