r"""Recursive type aliases for JSON-like, TOML-like, and YAML-like objects."""

__all__ = [
    "JSON",
    "TOML",
    "YAML",
]

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
