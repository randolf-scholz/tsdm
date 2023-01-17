"""Collection of Useful Type Aliases."""

__all__ = [
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
from typing import Any, Callable, TypeAlias

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.dtypes.base import ExtensionDtype

from tsdm.types.protocols import Lookup
from tsdm.types.variables import AnyVar as T
from tsdm.types.variables import Key_contra
from tsdm.types.variables import KeyVar as K
from tsdm.types.variables import Value_co
from tsdm.types.variables import ValueVar as V

# region Custom Type Aliases -----------------------------------------------------------
Map: TypeAlias = Lookup[Key_contra, Value_co] | Callable[[Key_contra], Value_co]
r"""Type Alias for `Map`."""
Nested: TypeAlias = T | Collection["Nested[T]"] | Mapping[Any, "Nested[T]"]
r"""Type Alias for nested types (JSON-Like)."""
PandasObject: TypeAlias = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""
PathLike: TypeAlias = str | Path | os.PathLike[str]
r"""Type Alias for path-like objects."""
ScalarDType: TypeAlias = type[np.generic] | torch.dtype | type[ExtensionDtype]  # type: ignore[index]
r"""TypeAlias for scalar types."""
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
JSON: TypeAlias = None | str | int | float | bool | list["JSON"] | dict[str, "JSON"]  # type: ignore[operator]
r"""Type Alias for JSON-Like objects."""
TOML: TypeAlias = None | str | int | float | bool | list["TOML"] | dict[str, "TOML"]  # type: ignore[operator]
r"""Type Alias for JSON-Like objects."""
YAML: TypeAlias = None | str | int | float | bool | list["YAML"] | dict[str, "YAML"]  # type: ignore[operator]
r"""Type Alias for JSON-Like objects."""
# endregion Nested Configuration ----------------------------------------------
