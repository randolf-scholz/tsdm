r"""Serialization utilities for tables."""

__all__ = [
    "serialize_table",
    "deserialize_table",
]

from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import IO, Any, Optional

import pandas as pd
from pyarrow import Table as PyArrowTable, parquet

from tsdm.types.aliases import FilePath


def serialize_table(
    table: Any,
    path_or_buf: FilePath | IO[bytes],
    /,
    *,
    writer: Optional[str | Callable] = None,
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
    match path_or_buf, writer:
        case [str() | PathLike() as filepath, _]:
            path = Path(filepath)
            if not path.suffix.startswith("."):
                raise ValueError("File must have a suffix!")
            writer = path.suffix[1:] if writer is None else writer
            target = path
        case [target, str() | Callable()]:  # type: ignore[misc]
            pass
        case _:
            raise TypeError(
                "Provide either a PathLike object or a buffer-like object with an extension!"
            )

    match writer, table:
        case [Callable() as writer_impl, _]:  # type: ignore[misc]
            writer_impl(target, **writer_kwargs)  # type: ignore[unreachable]
        case ["parquet", PyArrowTable() as arrow_table]:
            parquet.write_table(arrow_table, target, **writer_kwargs)
        case [str(extension), _] if hasattr(table, f"to_{extension}"):
            writer_impl = getattr(table, f"to_{extension}")
            writer_impl(target, **writer_kwargs)
        case _:
            raise NotImplementedError(f"No serializer implemented for {writer=}")


def deserialize_table(
    path_or_buf: FilePath | IO[bytes],
    /,
    *,
    loader: Optional[str | Callable] = None,
    **loader_kwargs: Any,
) -> Any:
    r"""Deserialize a table."""
    match path_or_buf, loader:
        case [str() | PathLike() as filepath, _]:
            path = Path(filepath)
            if not path.suffix.startswith("."):
                raise ValueError("File must have a suffix!")
            loader = path.suffix[1:] if loader is None else loader
            target = path
        case [target, str() | Callable()]:  # type: ignore[misc]
            pass
        case _:
            raise TypeError(
                "Provide either a PathLike object or a buffer-like object with an extension!"
            )

    match loader:
        case Callable() as loader_impl:  # type: ignore[misc]
            return loader_impl(target, **loader_kwargs)  # type: ignore[unreachable]
        case str(extension) if hasattr(pd, f"read_{extension}"):
            loader_impl = getattr(pd, f"read_{extension}")
            if extension in {"csv", "parquet", "feather"}:
                loader_kwargs = {
                    "engine": "pyarrow",
                    "dtype_backend": "pyarrow",
                } | dict(loader_kwargs)
            return loader_impl(target, **loader_kwargs).squeeze()
        case _:
            raise NotImplementedError(f"No loader available! {loader=}")
