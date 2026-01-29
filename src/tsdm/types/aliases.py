r"""Collection of Useful Type Aliases."""

__all__ = [
    # path types
    "DirPath",
    "FilePath",
    "PathLike",
    # argument types
    "DictArg",
    "IndexArg1D",
    "IndexArgND",
    "Axis",
    "DimArg",
    "Size",
    # JSON-like Types
    "JSON",
    "TOML",
    "YAML",
    # Nested ABCs
    "Nested",
    "NestedMapping",
    "NestedDict",
    "NestedBuiltin",
]


import os
from collections.abc import Collection, Iterable, Mapping
from datetime import datetime
from types import EllipsisType

# region function argument aliases -----------------------------------------------------
type PathLike = str | os.PathLike[str]
r"""Type Alias for path-like objects."""
type FilePath = str | os.PathLike[str]
r"""Type Alias for path-like objects pointing to file."""
type DirPath = str | os.PathLike[str]
r"""Type Alias for path-like objects pointing to directory."""
type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type DimArg = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""
type IndexArg1D = None | int | slice | range | list[int] | list[bool] | EllipsisType
r"""Type alias for `__getitem__` argument for tensors."""
type IndexArgND = IndexArg1D | tuple[IndexArg1D, ...]
r"""Indexer that always returns a sub-tensor."""
type DictArg[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
r"""Type Alias for dictionary-like arguments."""
# endregion function argument aliases --------------------------------------------------

# region generic type aliases ----------------------------------------------------------
type Nested[T] = Mapping[str, Nested[T]] | Collection[Nested[T]] | T  # +T
r"""Type Alias for nested types (JSON-Like)."""
type NestedMapping[K, V] = Mapping[K, V | NestedMapping[K, V]]
r"""Generic Type Alias for nested `Mapping`."""
type NestedDict[K, V] = dict[K, V | NestedDict[K, V]]
r"""Generic Type Alias for nested `dict`."""
type NestedBuiltin[T] = (
    T
    | tuple[T, ...]  # leaf-tuple
    | tuple[NestedBuiltin[T], ...]
    | set[T]  # leaf-set
    | set[NestedBuiltin[T]]
    | frozenset[T]  # leaf-frozenset
    | frozenset[NestedBuiltin[T]]
    | list[T]  # leaf-list
    | list[NestedBuiltin[T]]
    | dict[str, T]  # leaf-dict
    | dict[str, NestedBuiltin[T]]
)
r"""Type Alias for nested builtins."""
# endregion generic type aliases -------------------------------------------------------

# region JSON-like types ---------------------------------------------------------------
type JSON_LEAF = None | bool | int | float | str
type TOML_LEAF = None | bool | int | float | str | datetime
type YAML_LEAF = None | bool | int | float | str | datetime

type JSON = JSON_LEAF | list[JSON] | dict[str, JSON]
r"""Type Alias for JSON-Like objects."""
type YAML = YAML_LEAF | list[YAML] | dict[str, YAML]
r"""Type Alias for JSON-Like objects."""
type TOML = TOML_LEAF | list[TOML] | dict[str, TOML]
r"""Type Alias for JSON-Like objects."""
# endregion JSON-like types ------------------------------------------------------------
