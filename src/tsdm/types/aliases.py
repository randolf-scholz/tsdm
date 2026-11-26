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
    # nested
    "JSON",
    "TOML",
    "YAML",
]

import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from types import EllipsisType
from typing import IO

type FilePath = str | os.PathLike[str]  # includes pathlib.Path
r"""Type alias for strings and path-like objects identifying files."""
type FileStream = IO[bytes]  # Note: io.BytesIO <: IO[bytes]
r"""Type alias for binary file-like objects."""
type Axis = None | int | tuple[int, ...]
r"""Type alias for a single axis, a tuple of axes, or `None` for all axes."""
type DictArg[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
r"""Type alias for mappings or iterables of key-value pairs."""
type IndexArg1D = None | int | slice | range | list[int] | list[bool] | EllipsisType
r"""Type alias for a tensor index along a single dimension."""
type IndexArgND = IndexArg1D | tuple[IndexArg1D, ...]
r"""Type alias for a single tensor index or a tuple of indices."""


type _JSON_LEAF = None | bool | int | float | str
type _TOML_LEAF = None | bool | int | float | str | datetime
type _YAML_LEAF = None | bool | int | float | str | datetime

type JSON = _JSON_LEAF | list[JSON] | dict[str, JSON]
r"""Type alias for JSON values with recursively nested lists and string-keyed dictionaries."""
type YAML = _YAML_LEAF | list[YAML] | dict[str, YAML]
r"""Type alias for YAML-like values with nested lists and string-keyed dictionaries.

This alias models a subset of YAML values, including `datetime` leaves.
"""
type TOML = _TOML_LEAF | list[TOML] | dict[str, TOML]
r"""Type alias for TOML-like values with nested lists and string-keyed dictionaries.

This in-memory representation includes `None`; it does not enforce TOML's
serialization constraints.
"""
