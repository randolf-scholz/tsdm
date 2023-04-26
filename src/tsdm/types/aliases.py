"""Collection of Useful Type Aliases."""

__all__ = [
    # namedtuple
    "schema",
    # Custom Type Aliases
    "Nested",
    "Map",
    "PandasObject",
    "PathLike",
    "ScalarDType",
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
    # Nested Builtins
    "NestedList",
    "NestedDict",
    "NestedSet",
    "NestedTuple",
    "NestedFrozenSet",
    "NestedMapping",
    "NestedSequence",
]


import os
from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, TypeAlias

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.dtypes.base import ExtensionDtype

from tsdm.types.protocols import Lookup
from tsdm.types.variables import any_var as T
from tsdm.types.variables import key_contra
from tsdm.types.variables import key_var as K
from tsdm.types.variables import value_co
from tsdm.types.variables import value_var as V


class schema(NamedTuple):
    """Table schema."""

    shape: Optional[tuple[int, int]] = None
    """Shape of the table."""
    columns: Optional[Sequence[str]] = None
    """Column names of the table."""
    dtypes: Optional[Sequence[str]] = None
    """Data types of the columns."""


# region Custom Type Aliases -----------------------------------------------------------
Map: TypeAlias = Lookup[key_contra, value_co] | Callable[[key_contra], value_co]
r"""Type Alias for `Map`."""
Nested: TypeAlias = T | Collection["Nested[T]"] | Mapping[Any, "Nested[T]"]
r"""Type Alias for nested types (JSON-Like)."""
PandasObject: TypeAlias = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""
PathLike: TypeAlias = str | Path | os.PathLike[str]
r"""Type Alias for path-like objects."""
ScalarDType: TypeAlias = type[np.generic] | torch.dtype | type[ExtensionDtype]
r"""TypeAlias for scalar types."""
ContainerLike: TypeAlias = T | Lookup[int, T] | Callable[[int], T]
r"""Type Alias for container-like objects."""
# endregion Custom Type Aliases --------------------------------------------------------


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
# endregion Nested Builtins ------------------------------------------------------------


# region Nested Configuration -------------------------------------------------
JSON: TypeAlias = None | str | int | float | bool | list["JSON"] | dict[str, "JSON"]
r"""Type Alias for JSON-Like objects."""
TOML: TypeAlias = None | str | int | float | bool | list["TOML"] | dict[str, "TOML"]
r"""Type Alias for JSON-Like objects."""
YAML: TypeAlias = None | str | int | float | bool | list["YAML"] | dict[str, "YAML"]
r"""Type Alias for JSON-Like objects."""


# endregion Nested Configuration ----------------------------------------------
