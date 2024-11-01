r"""Serialization utilities for tables."""

__all__ = [
    "serialize_table",
    "deserialize_table",
]

from collections.abc import Callable as Fn
from pathlib import Path
from typing import IO, Any, Concatenate, Optional

import pandas as pd
import polars as pl
import pyarrow as pa
from pyarrow import parquet as pyarrow_parquet

from tsdm.types.aliases import FilePath


def serialize_table[T](
    table: T,
    path_or_buf: FilePath | IO[bytes],
    /,
    *,
    writer: Optional[str | Fn[Concatenate[T, ...], None]] = None,
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
        case None:
            try:
                path = Path(path_or_buf)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("Cannot determine writer") from None
            return serialize_table(table, path, writer=path.suffix[1:], **writer_kwargs)
        case writer_impl if callable(writer_impl):
            return writer_impl(table, path_or_buf, **writer_kwargs)
        case str(ext):
            pass
        case _:
            raise TypeError(f"Invalid writer: {writer=}")

    match table, ext:
        case pd.DataFrame(), _:
            writer_method = getattr(table, f"to_{ext}")
            writer_method(table, path_or_buf, **writer_kwargs)
        case pl.DataFrame(), _:
            writer_method = getattr(table, f"write_{ext}")
            writer_method(table, path_or_buf, **writer_kwargs)
        case pa.Table(), "parquet":
            pyarrow_parquet.write_table(table, path_or_buf, **writer_kwargs)
        case _ if callable(writer_method := getattr(table, f"to_{ext}", None)):
            writer_method(table, path_or_buf, **writer_kwargs)
        case _:
            raise NotImplementedError(f"No serializer implemented for {writer=}")


def deserialize_table(
    path_or_buf: FilePath | IO[bytes],
    /,
    *,
    loader: Optional[str | Fn] = None,
    **loader_kwargs: Any,
) -> Any:
    r"""Deserialize a table."""
    match loader:
        case None:
            try:
                path = Path(path_or_buf)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("Cannot determine loader") from None
            return deserialize_table(path, loader=path.suffix[1:], **loader_kwargs)
        case loader_impl if callable(loader_impl):
            return loader_impl(path_or_buf, **loader_kwargs)
        case str(ext) if callable(loader_method := getattr(pd, f"read_{ext}", None)):
            loader_kwargs = {
                "engine": "pyarrow",
                "dtype_backend": "pyarrow",
            } | loader_kwargs
            df: pd.DataFrame = loader_method(path_or_buf, **loader_kwargs)
            return df.squeeze()
        case _:
            raise NotImplementedError(f"No loader implemented for {loader=}")
