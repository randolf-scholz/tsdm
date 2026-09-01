r"""Nested/Recursive Type Aliases for JSON, TOML, and YAML-like objects."""

__all__ = [
    "JSON",
    "TOML",
    "YAML",
    # Nested ABCs
    "Nested",
    "NestedMapping",
    "NestedDict",
    "NestedBuiltin",
]

from collections.abc import Collection, Mapping
from datetime import datetime

type JSON_LEAF = None | bool | int | float | str
type TOML_LEAF = None | bool | int | float | str | datetime
type YAML_LEAF = None | bool | int | float | str | datetime

type JSON = JSON_LEAF | list[JSON] | dict[str, JSON]
r"""Type Alias for JSON-Like objects."""
type YAML = YAML_LEAF | list[YAML] | dict[str, YAML]
r"""Type Alias for JSON-Like objects."""
type TOML = TOML_LEAF | list[TOML] | dict[str, TOML]
r"""Type Alias for JSON-Like objects."""


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
