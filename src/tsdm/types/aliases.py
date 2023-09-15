"""Collection of Useful Type Aliases."""

__all__ = [
    # namedtuple
    # Custom Type Aliases
    "Axes",
    "Nested",
    "Map",
    "PandasObject",
    "PathLike",
    "ScalarDType",
    "SizeLike",
    # Maybe Type Aliases
    "MaybeCallable",
    "MaybeFrozenset",
    "MaybeIterable",
    "MaybeList",
    "MaybeSet",
    "MaybeTuple",
    # Configuration
    "JSON",
    "TOML",
    "YAML",
    # Nested ABCs
    "NestedCollection",
    "NestedIterable",
    "NestedMapping",
    "NestedMutableMapping",
    "NestedSequence",
    "NestedGeneric",
    # Nested Builtins
    "NestedList",
    "NestedDict",
    "NestedSet",
    "NestedTuple",
    "NestedFrozenSet",
    "NestedBuiltin",
]


import os
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.dtypes.base import ExtensionDtype

from tsdm.types.protocols import Lookup
from tsdm.types.variables import (
    any_co as T_co,
    any_var as T,
    key_contra,
    key_var as K,
    value_co,
    value_var as V,
)

# region Custom Type Aliases -----------------------------------------------------------
PythonScalar: TypeAlias = (
    None | bool | int | float | complex | str | datetime | timedelta
)
r"""Type Alias for Python scalars."""

Axes: TypeAlias = None | int | tuple[int, ...]
r"""Type Alias for axes."""
SizeLike: TypeAlias = int | tuple[int, ...]
r"""Type Alias for shape-like objects."""

Map: TypeAlias = Lookup[key_contra, value_co] | Callable[[key_contra], value_co]
r"""Type Alias for `Map`."""
Nested: TypeAlias = T_co | Collection["Nested[T_co]"] | Mapping[Any, "Nested[T_co]"]
r"""Type Alias for nested types (JSON-Like)."""
PandasObject: TypeAlias = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""
PathLike: TypeAlias = str | Path | os.PathLike[str]
r"""Type Alias for path-like objects."""
ScalarDType: TypeAlias = type[np.generic] | torch.dtype | type[ExtensionDtype]
r"""TypeAlias for scalar types."""
ContainerLike: TypeAlias = T_co | Lookup[int, T_co] | Callable[[int], T_co]
r"""Type Alias for container-like objects."""
# endregion Custom Type Aliases --------------------------------------------------------


# region Maybe Type Aliases ------------------------------------------------------------
# NOTE: Maybe refers to types of the kind Union[T, Container[T]]
MaybeList: TypeAlias = T | list[T]
r"""Type Alias for T | list[T]."""
MaybeTuple: TypeAlias = T | tuple[T, ...]
r"""Type Alias for T | tuple[T, ...]."""
MaybeFrozenset: TypeAlias = T | frozenset[T]
r"""Type Alias for T | frozenset[T]."""
MaybeSet: TypeAlias = T | set[T]
r"""Type Alias for T | set[T]."""
MaybeIterable: TypeAlias = T | Iterable[T]
r"""Type Alias for T | Iterable[T]."""
MaybeCallable: TypeAlias = T | Callable[[], T]
r"""Type Alias for objects that maybe needs to be created first."""
# endregion Maybe Type Aliases ---------------------------------------------------------


# region Nested collections.abc --------------------------------------------------------
NestedCollection: TypeAlias = Collection[V | "NestedCollection[V]"]
r"""Generic Type Alias for nested `Collection`."""
NestedIterable: TypeAlias = Iterable[V | "NestedIterable[V]"]
r"""Generic Type Alias for nested `Iterable`."""
NestedMapping: TypeAlias = Mapping[K, V | "NestedMapping[K,V]"]
r"""Generic Type Alias for nested `Mapping`."""
NestedMutableMapping: TypeAlias = MutableMapping[K, V | "NestedMutableMapping[K, V]"]
r"""Generic Type Alias for nested `MutableMapping`."""
NestedSequence: TypeAlias = Sequence[V | "NestedSequence[V]"]
r"""Generic Type Alias for nested `Sequence`."""
NestedGeneric: TypeAlias = (
    Sequence[V | "NestedGeneric[V]"] | Mapping[str, V | "NestedGeneric[V]"]
)
r"""Generic Type Alias for nested `Sequence` or `Mapping`."""
# endregion Nested collections.abc -----------------------------------------------------


# region Nested Builtins ---------------------------------------------------------------
NestedDict: TypeAlias = dict[K, V | "NestedDict[K, V]"]
r"""Generic Type Alias for nested `dict`."""
NestedList: TypeAlias = list[V | "NestedList[V]"]
r"""GenericType Alias for nested `list`."""
NestedSet: TypeAlias = set[V | "NestedSet[V]"]
r"""Generic Type Alias for nested `set`."""
NestedFrozenSet: TypeAlias = frozenset[V | "NestedFrozenSet[V]"]
r"""Generic Type Alias for nested `set`."""
NestedTuple: TypeAlias = tuple[V | "NestedTuple[V]", ...]
r"""Generic Type Alias for nested `tuple`."""
NestedBuiltin: TypeAlias = (
    tuple[V | "NestedBuiltin[V]", ...]
    | set[V | "NestedBuiltin[V]"]
    | frozenset[V | "NestedBuiltin[V]"]
    | list[V | "NestedBuiltin[V]"]
    | dict[str, V | "NestedBuiltin[V]"]
)
# endregion Nested Builtins ------------------------------------------------------------


# region Nested Configuration ----------------------------------------------------------
JSON_LeafType: TypeAlias = None | bool | int | float | str
TOML_LeafType: TypeAlias = None | bool | int | float | str | datetime
YAML_LeafType: TypeAlias = None | bool | int | float | str | datetime

JSON: TypeAlias = JSON_LeafType | list["JSON"] | dict[str, "JSON"]
r"""Type Alias for JSON-Like objects."""
TOML: TypeAlias = TOML_LeafType | list["TOML"] | dict[str, "TOML"]
r"""Type Alias for JSON-Like objects."""
YAML: TypeAlias = YAML_LeafType | list["YAML"] | dict[str, "YAML"]
r"""Type Alias for JSON-Like objects."""
# endregion Nested Configuration -------------------------------------------------------
