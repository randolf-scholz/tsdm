r"""Common type aliases for paths, shapes, mappings, and tensor indices."""

__all__ = [
    # path types
    "FilePath",
    "FileStream",
    # argument types
    "DictArg",
    "Axis",
    "IndexArg1D",
    "IndexArgND",
]

import os
from collections.abc import Iterable, Mapping
from io import BytesIO
from types import EllipsisType
from typing import IO

type FilePath = str | os.PathLike[str]
r"""Type alias for strings and path-like objects identifying files."""
type FileStream = IO[bytes] | BytesIO
r"""Type alias for binary file-like objects."""
type Axis = None | int | tuple[int, ...]
r"""Type alias for a single axis, a tuple of axes, or `None` for all axes."""
type DictArg[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
r"""Type alias for mappings or iterables of key-value pairs."""
type IndexArg1D = None | int | slice | range | list[int] | list[bool] | EllipsisType
r"""Type alias for a tensor index along a single dimension."""
type IndexArgND = IndexArg1D | tuple[IndexArg1D, ...]
r"""Type alias for a single tensor index or a tuple of indices."""
