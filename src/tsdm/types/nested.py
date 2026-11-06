r"""Recursive type aliases for JSON-like, TOML-like, and YAML-like objects."""

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
r"""Type alias for JSON values with recursively nested lists and string-keyed dictionaries."""
type YAML = YAML_LEAF | list[YAML] | dict[str, YAML]
r"""Type alias for YAML-like values with nested lists and string-keyed dictionaries.

This alias models a subset of YAML values, including `datetime` leaves.
"""
type TOML = TOML_LEAF | list[TOML] | dict[str, TOML]
r"""Type alias for TOML-like values with nested lists and string-keyed dictionaries.

This in-memory representation includes `None`; it does not enforce TOML's
serialization constraints.
"""


type Nested[T] = Mapping[str, Nested[T]] | Collection[Nested[T]] | T  # +T
r"""Type alias for leaves of type `T` nested in string-keyed mappings or collections."""
type NestedMapping[K, V] = Mapping[K, V | NestedMapping[K, V]]
r"""Type alias for mappings whose values are leaves or recursively nested mappings."""
type NestedDict[K, V] = dict[K, V | NestedDict[K, V]]
r"""Type alias for dictionaries whose values are leaves or recursively nested dictionaries."""
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
r"""Type alias for leaves nested in tuples, sets, frozensets, lists, or dictionaries.

Dictionary keys are strings. Values stored in sets or frozensets must be hashable.
"""
