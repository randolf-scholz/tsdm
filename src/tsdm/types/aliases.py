r"""Collection of Useful Type Aliases."""

__all__ = [
    # path types
    "FilePath",
    "FileStream",
    # argument types
    "DictArg",
    "Axis",
    "Size",
    "IndexArg1D",
    "IndexArgND",
]

import os
from collections.abc import Iterable, Mapping
from io import BytesIO
from types import EllipsisType
from typing import IO

type FilePath = str | os.PathLike[str]
r"""Type Alias for path-like objects pointing to file."""
type FileStream = IO[bytes] | BytesIO
r"""Type Alias for file-like objects."""
type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type DictArg[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
r"""Type Alias for dictionary-like arguments."""
type IndexArg1D = None | int | slice | range | list[int] | list[bool] | EllipsisType
r"""Type alias for `__getitem__` argument for tensors."""
type IndexArgND = IndexArg1D | tuple[IndexArg1D, ...]
r"""Indexer that always returns a sub-tensor."""
