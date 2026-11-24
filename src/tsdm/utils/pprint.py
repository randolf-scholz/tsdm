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
    r"""Format known scalars without calling subclass hooks, or return `None`."""
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
    if issubclass(cls, dt.datetime):
        value = cast("dt.datetime", arg)
        # Even the base formatter invokes repr() on the stored timezone.
        if dt.datetime.tzinfo.__get__(value) is not None:
            return None
        return dt.datetime.__repr__(value)
    if issubclass(cls, dt.timedelta):
        return dt.timedelta.__repr__(cast("dt.timedelta", arg))
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


def _get_dtypes(arg: SupportsDataFrame | HasDTypes, /) -> list[object]:
    r"""Return column dtypes from native metadata or dataframe interchange."""
    match arg:
        case object(dtypes=dtypes):  # type: ignore
            return list(dtypes)
        case SupportsDataFrame():
            frame = arg.__dataframe__()
            # A protocol dtype is (kind, bit width, format string, endianness).
            dtypes = (column.dtype for column in frame.get_columns())
            return [f"{kind.name}{width}" for kind, width, _, _ in dtypes]
        case _:
            raise TypeError


def _get_dtype(arg: SupportsArray, /) -> object:
    r"""Return the dtype of a homogeneous array."""
    match arg:
        case object(dtype=dtype):  # type: ignore
            return dtype
        case object(type=dtype):  # type: ignore
            # pa.Array
            return dtype
        case _:
            return arg.__array__().dtype


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

    linebreaks = (
        False if size == 0  # always inline if empty mapping
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

    left, right = "(", ")"

    # special treatment for dict:
    if cls is dict:
        left, right = "{", "}"
        title = ""
    elif identifier is None and recursive:
        identifier = Identifier.get(wrapped)

    # set repr_func
    repr_func = (
        repr_fn if repr_fn is not None else repr_object if recursive else repr_compact
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
    head_end, tail_start = _item_bounds(size, maxitems)
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
        linebreaks=linebreaks,
        padding=padding,
    )

    # assemble the string and add indentation
    string = f"{title}{modifier}{Identifier.render(identifier)}{left}{body}{right}"
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

    linebreaks = (
        False if size == 0  # always inline if empty sequence
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    maxitems = (
        maxitems if maxitems is not None
        else MAXITEMS if linebreaks
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
    repr_func = (
        repr_fn if repr_fn is not None else repr_object if recursive else repr_compact
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
    head_end, tail_start = _item_bounds(size, maxitems)
    head = [to_string(i, value) for i, value in enumerate(obj[:head_end])]
    tail = [to_string(i, value) for i, value in enumerate(obj[tail_start:], tail_start)]
    body = _join_items(
        head,
        tail,
        truncated=head_end < tail_start,
        linebreaks=linebreaks,
        padding=padding,
    )

    # assemble the string and add indentation
    string = f"{title}{modifier}{Identifier.render(identifier)}{left}{body}{right}"
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

    linebreaks = (
        False if size == 0  # always inline if empty set
        else linebreaks if linebreaks is not None
        else bool(recursive)  # fallback
    )  # fmt: skip

    maxitems = (
        maxitems if maxitems is not None
        else MAXITEMS if linebreaks
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
    repr_func = (
        repr_fn if repr_fn is not None else repr_object if recursive else repr_compact
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
    head_end, tail_start = _item_bounds(size, maxitems)
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
        linebreaks=linebreaks,
        padding=padding,
    )

    # assemble the string and add indentation
    string = f"{title}{modifier}{Identifier.render(identifier)}{left}{body}{right}"
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
    dtypes = _get_dtypes(obj)

    match obj:
        case object(shape=tuple(sizes)):  # type: ignore
            num_rows, num_cols = sizes
            assert num_cols == len(dtypes)
        case _:
            num_rows = len(obj) if isinstance(obj, Sized) else None
            num_cols = len(dtypes)
    shape = f"{'*' if num_rows is None else num_rows},{num_cols}"

    maxitems = MAXITEMS_INLINE if maxitems is None else maxitems
    labels = [_repr_dtype(dtype) for dtype in dtypes]
    if len(set(labels)) <= 1:
        labels = labels[:1]
    head_end, tail_start = _item_bounds(len(labels), maxitems)
    body = _join_items(
        labels[:head_end], labels[tail_start:], truncated=head_end < tail_start
    )

    if modifier is None and show_device and isinstance(obj, HasDevice):
        modifier = str(obj.device)
    title = type(obj).__name__ if title is None else title
    modifier_repr = f"@{modifier}" if modifier is not None else ""
    shape_repr = f"<{shape}>" if identifier is None else Identifier.render(identifier)
    return f"{title}{modifier_repr}{shape_repr}[{body}]"


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
    r"""Summarize a homogeneous array by shape, dtype, or its only value."""
    type_repr = type(obj).__name__ if title is None else title
    shape: tuple[int, ...] = (
        tuple(obj.shape) if isinstance(obj, HasShape) else tuple(obj.__array__().shape)
    )
    if identifier is None:
        dimensions = ",".join(str(dim) for dim in shape)
        shape_repr = f"<{dimensions}>" if dimensions else ""
    else:
        shape_repr = Identifier.render(identifier)

    if not show_device:
        device_repr = ""
    else:
        match modifier, obj:
            case None, HasDevice(device=device):
                device_repr = f"@{device}"
            case _:
                device_repr = f"@{modifier}" if modifier is not None else ""

    match prod(shape):
        case 0:
            suffix = ""
        case 1:
            value = (
                obj.item() if isinstance(obj, SupportsItem) else obj.__array__().item()
            )
            suffix = f"({value!s})"
        case _:
            suffix = f"[{_repr_dtype(_get_dtype(obj))}]"
    return f"{type_repr}{device_repr}{shape_repr}{suffix}"


def _repr_type(obj: type, /, *, identifier: str | None = None, **_: Any) -> str:
    r"""Return a compact representation of a type."""
    return f"{obj.__name__}{Identifier.render(identifier)}"


def _repr_function(
    obj: FunctionType | BuiltinFunctionType,
    /,
    *,
    identifier: str | None = None,
    **_: Any,
) -> str:
    r"""Format a Python or built-in function with any supplied identifier."""
    label = Identifier.render(identifier)
    if type(obj) is BuiltinFunctionType:
        return f"{label}{obj!r}"
    return f"{obj.__name__}{label}"


def repr_compact(
    obj: object,
    /,
    *,
    identifier: str | None = None,
    recursive: bool | int = False,
    wrapped: object | None = None,
    **_: Any,
) -> str:
    r"""Return a compact representation using a value or its type name."""
    obj_type = type(obj)
    for entry in (Identifier.TYPE, Identifier.FUNCTION):
        if issubclass(obj_type, entry.kind):
            return entry.formatter(obj, identifier=identifier)

    label = Identifier.render(identifier)
    if (scalar_repr := _repr_python_scalar(obj)) is not None:
        return f"{label}{scalar_repr}"
    if (number_repr := _repr_number(obj)) is not None:
        return f"{label}{number_repr}"

    displayed = obj if wrapped is None else wrapped
    if identifier is None and recursive:
        label = Identifier.render(Identifier.get(displayed))
    return f"{type(displayed).__name__}{label}()"


def repr_object(
    obj: object,
    /,
    *,
    identifier: str | None = None,
    fallback: Callable[..., str] = repr_compact,
    **kwargs: Any,
) -> str:
    r"""Return a string representation of an object.

    Dispatch to specialized formatters for scalars, arrays, containers,
    dataclasses, named tuples, and dtypes. Use `fallback` for unsupported objects.
    A custom `identifier` must satisfy `str.isidentifier()` and is displayed
    in angle brackets. An empty string suppresses the identifier. Container
    formatters use the type of `wrapped` for automatic identifiers when a
    wrapper is supplied.

    For containers and table dtype summaries, `maxitems` is a nonnegative
    limit on displayed entries, excluding the ellipsis. When entries are
    omitted, an odd limit gives the head one more entry than the tail.

    Examples:
        ```pycon
        >>> repr_object(1)
        '1'
        >>> repr_object(3.14)
        '3.14'

        ```
    """
    label = Identifier.render(identifier)
    if identifier is not None:
        kwargs["identifier"] = identifier

    if (scalar_repr := _repr_python_scalar(obj)) is not None:
        return f"{label}{scalar_repr}"
    if (entry := Identifier.get(obj)) is not None:
        return entry.formatter(obj, **kwargs)

    if fallback is repr_compact:
        return repr_compact(obj, **kwargs)
    # Avoid calling an unsupported object's repr method again.
    if fallback is repr:
        return f"{label}{object.__repr__(obj)}"
    return f"{label}{fallback(obj)}"


class Identifier(StrEnum):
    r"""Labels, runtime types, and formatters in dispatch order."""

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

    def matches(self, arg: object, /) -> bool:
        r"""Check whether this category applies to the object."""
        if self is Identifier.SEQUENCE and isinstance(arg, str):
            return False
        if isinstance(arg, self.kind):
            return True
        # Tables can expose native column dtypes without interchange.
        return (
            self is Identifier.TABLE
            and isinstance(arg, HasDTypes)
            and not isinstance(arg, HasDType)
            and (not isinstance(arg, HasShape) or len(arg.shape) == 2)
        )

    @classmethod
    def get(cls, arg: object, /) -> Identifier | None:
        r"""Select the first applicable category."""
        return next((entry for entry in Identifier if entry.matches(arg)), None)

    @classmethod
    def _missing_(cls, value: object) -> Identifier | None:
        return None if isinstance(value, str) else Identifier.get(value)

    @staticmethod
    def render(identifier: str | None, /) -> str:
        r"""Enclose a valid identifier in angle brackets, or omit an empty label."""
        if not identifier:
            return ""
        if not str.isidentifier(identifier):
            raise ValueError(
                f"Invalid identifier {identifier!r}. Use a Python identifier."
            )
        return f"<{identifier}>"

    def validate(self, arg: object) -> None:
        r"""Raise an assertion error when this category does not apply."""
        if self is Identifier.SEQUENCE and isinstance(arg, str):
            raise AssertionError("The <sequence> identifier excludes strings.")
        if not self.matches(arg):
            raise AssertionError(
                f"<{self.value}> does not apply to {type(arg).__name__}."
            )


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
    r"""Install a `__repr__` selected from the class's supported object category.

    Forward `kwds` to the selected formatter. Return the modified class, or
    raise `TypeError` for unsupported classes and nonclass inputs.
    """
    if not isinstance(cls, type):
        raise TypeError("Must be a class!")

    repr_func: Callable[..., str]

    if is_dataclass(cls):
        repr_func = _repr_dataclass
    elif is_namedtuple(cls):
        repr_func = _repr_namedtuple
    elif issubclass(cls, Mapping):
        repr_func = _repr_mapping
    elif (
        issubclass(cls, (SupportsArray, SupportsDataFrame))  # pyrefly: ignore[bad-argument-type]
        or isinstance(cls, HasDTypes)
    ):
        # Some tables can only be distinguished from arrays through instance metadata.
        repr_func = repr_object
    elif issubclass(cls, Sequence):  # pyrefly: ignore[bad-argument-type]
        repr_func = _repr_sequence
    elif issubclass(cls, AbstractSet):  # pyrefly: ignore[bad-argument-type]
        repr_func = _repr_set
    elif issubclass(cls, type):  # pyrefly: ignore[bad-argument-type]
        repr_func = repr_compact
    else:
        raise TypeError(f"Unsupported type {cls}.")

    cls.__repr__ = partial(repr_func, **kwds)  # ty: ignore[invalid-assignment]
    return cls
