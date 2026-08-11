r"""Serialization utilities for tables."""

__all__ = [
    "InlineTable",
    "serialize_table",
    "deserialize_table",
    "make_dataframe",
]

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Concatenate, NotRequired, Optional, Required, TypedDict

import pandas as pd
import polars as pl
import pyarrow as pa
from pyarrow import parquet as pyarrow_parquet

from tsdm.types.aliases import FilePath, FileStream

type Writer[T] = Callable[Concatenate[T, ...], None]
type Loader[T] = Callable[Concatenate[FilePath | FileStream, ...], T]


def _choose_default_writer[T: pa.Table | pd.DataFrame | pl.DataFrame](
    table: T, extension: str
) -> Writer[T]:
    r"""Default writer function that uses the extension of the path."""
    match table, extension:
        case pa.Table(), "parquet":
            return pyarrow_parquet.write_table
        case pd.DataFrame() as pd_frame, ext:
            return getattr(type(pd_frame), f"to_{ext}")
        case pl.DataFrame() as pl_frame, ext:
            return getattr(type(pl_frame), f"write_{ext}")
        case _:
            if (writer := getattr(type(table), f"to_{extension}", None)) is not None:
                return writer
            raise NotImplementedError(
                f"Unsupported extension/table {extension}/{type(table)}"
                "\n Please provide a writer function."
            )


def serialize_table[T](
    table: T,
    path_or_buf: FilePath | FileStream,
    /,
    *,
    writer: Optional[str | Writer[T]] = None,
    **writer_kwargs: Any,
) -> None:
    r"""Serialize a table.

    Args:
        table: Table to serialize.
        path_or_buf: Path or buffer to write to.
        writer: Writer to use.
            If None, the extension of the path is used to determine the writer.
            Pass string like 'csv' to use the corresponding `pandas` writer.
        **writer_kwargs: Additional keyword arguments to pass to the writer.
    """
    match writer:
        # call the provided writer function
        case writer_impl if callable(writer_impl):
            writer_impl(table, path_or_buf, **writer_kwargs)

        # use default write for the extension
        case str(extension):
            writer = _choose_default_writer(table, extension)
            writer(table, path_or_buf, **writer_kwargs)

        # try to determine the extension from the path
        case None:
            try:
                path = Path(path_or_buf)  # type: ignore
            except Exception as exc:
                exc.add_note(f"Cannot determine writer from {path_or_buf}")
                raise
            serialize_table(table, path, writer=path.suffix[1:], **writer_kwargs)
        case _:
            raise TypeError(f"Invalid writer: {type(writer)=}")


def _choose_default_loader(extension: str, /) -> Loader[Any]:
    r"""Default loader function that uses the extension of the path."""
    if (loader := getattr(pd, f"read_{extension}", None)) is not None:
        return loader

    raise NotImplementedError(f"Unsupported extension {extension}")


def deserialize_table[T = pd.DataFrame](
    path_or_buf: FilePath | FileStream,
    /,
    *,
    loader: Optional[str | Loader[T]] = None,
    **loader_kwargs: Any,
) -> T:
    r"""Deserialize a table."""
    match loader:
        # call the provided loader function
        case loader_impl if callable(loader_impl):
            return loader_impl(path_or_buf, **loader_kwargs)

        # use default loader for the extension
        case str(extension):
            loader_impl = _choose_default_loader(extension)
            return loader_impl(path_or_buf, **loader_kwargs)

        # determine the extension from the path
        case None:
            try:
                path = Path(path_or_buf)  # type: ignore
            except Exception as exc:
                exc.add_note(f"Cannot determine loader from {path_or_buf}")
                raise
            return deserialize_table(path, loader=path.suffix[1:], **loader_kwargs)

        case _:
            raise TypeError(f"Invalid loader type: {type(loader)=}")


class InlineTable[*Ts](TypedDict):
    r"""A table of data in a dictionary."""

    data: Required[Sequence[tuple[*Ts]]]
    schema: Mapping[str, Any]
    index: NotRequired[str | list[str]]


def make_dataframe(
    data: Sequence[tuple[Any, ...]],
    *,
    columns: Optional[list[str]] = None,
    dtypes: Optional[list[str | type]] = None,
    schema: Optional[Mapping[str, Any]] = None,
    index: Optional[str | list[str]] = None,
) -> pd.DataFrame:
    r"""Make a DataFrame from a dictionary."""
    if dtypes is not None and schema is not None:
        raise ValueError("Cannot specify both dtypes and schema.")

    if columns is not None:
        cols = list(columns)
    elif schema is not None:
        cols = list(schema)
    else:
        cols = None

    df = pd.DataFrame.from_records(data, columns=cols)

    if dtypes is not None:
        df = df.astype(dict(zip(df.columns, dtypes, strict=True)))
    elif schema is not None:
        df = df.astype(schema)

    if index is not None:
        df = df.set_index(index)

    return df
