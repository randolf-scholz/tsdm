r"""Collection of Useful Type Aliases."""

__all__ = [
    # New Types
    "SplitID",
    # namedtuple
    # Custom Type Aliases
    "Axis",
    "Dims",
    "FilePath",
    "Kwargs",
    "Map",
    "Nested",
    "PandasObject",
    "Shape",
    "Size",
    # Dtype Aliases
    "NumpyDtype",
    "NumpyDtypeArg",
    "PandasDtype",
    "PandasDTypeArg",
    "DType",
    # Scalar Type Aliases
    "Scalar",
    "StringScalar",
    "NumericalScalar",
    "TorchScalar",
    "TimeScalar",
    "PythonScalar",
    # Maybe Type Aliases
    "MaybeCallable",
    "MaybeFrozenset",
    "MaybeIterable",
    "MaybeList",
    "MaybeSet",
    "MaybeTuple",
    "MaybeWrapped",
    # Configuration
    "JSON",
    "TOML",
    "YAML",
    # Nested ABCs
    "NestedCollection",
    "NestedGeneric",
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
    "NestedBuiltin",
]


import os
from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.core.dtypes.base import ExtensionDtype
from typing_extensions import Any, TypeAlias

from tsdm.types.protocols import Lookup, SupportsKwargs
from tsdm.types.variables import K, K_contra, T, T_co, V, V_co

# region Custom Type Aliases -----------------------------------------------------------
Axis: TypeAlias = None | int | tuple[int, ...]
r"""Type Alias for axes."""
Dims: TypeAlias = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""  # FIXME: https://github.com/pytorch/pytorch/issues/64700
Size: TypeAlias = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
Shape: TypeAlias = int | tuple[int, ...]
r"""Type Alias for shape-like objects (note: `ones(shape=None)` creates 0d-array."""

Kwargs: TypeAlias = SupportsKwargs[T_co]
r"""Type Alias for keyword arguments."""

Map: TypeAlias = Lookup[K_contra, V_co] | Callable[[K_contra], V_co]
r"""Type Alias for `Map`."""
Nested: TypeAlias = T_co | Collection["Nested[T_co]"] | Mapping[Any, "Nested[T_co]"]
r"""Type Alias for nested types (JSON-Like)."""

FilePath: TypeAlias = str | Path | os.PathLike[str]  # cf. pandas._typing.FilePath
r"""Type Alias for path-like objects."""

ContainerLike: TypeAlias = T_co | Lookup[int, T_co] | Callable[[int], T_co]
r"""Type Alias for container-like objects."""
SplitID: TypeAlias = Hashable
r"""Type Alias for split identifiers."""
# endregion Custom Type Aliases --------------------------------------------------------


# region Dtype Aliases -----------------------------------------------------------------
NumpyDtype: TypeAlias = np.dtype
r"""Type Alias for `numpy` dtypes."""
NumpyDtypeArg: TypeAlias = str | type | NumpyDtype
r"""Type Alias for `numpy` dtype arguments."""
PandasDtype: TypeAlias = ExtensionDtype | NumpyDtype
r"""Type Alias for `pandas` dtype."""
PandasDTypeArg: TypeAlias = str | type | PandasDtype
r"""Type Alias for `pandas` dtype arguments."""
DType: TypeAlias = np.dtype | torch.dtype | type[ExtensionDtype]
r"""TypeAlias for dtypes."""
DTypeArg: TypeAlias = str | type
r"""TypeAlias for dtype arguments."""
PandasObject: TypeAlias = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""
# endregion Dtype Aliases --------------------------------------------------------------


# region Scalar Type Aliases -----------------------------------------------------------
Scalar: TypeAlias = bool | int | float | complex | str | bytes
r"""Type Alias for scalars."""
StringScalar: TypeAlias = str | bytes
r"""Type Alias for string scalars."""
NumericalScalar: TypeAlias = bool | int | float | complex
r"""Type Alias for numerical scalars."""
TorchScalar: TypeAlias = bool | int | float | str
r"""Type Alias for scalars allowed by torchscript."""
TimeScalar: TypeAlias = datetime | timedelta
r"""Type Alias for time scalars."""
PythonScalar: TypeAlias = NumericalScalar | StringScalar | TimeScalar
r"""Type Alias for Python scalars."""
# endregion Scalar Type Aliases --------------------------------------------------------


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
MaybeWrapped: TypeAlias = T_co | Callable[[], T_co] | Callable[[int], T_co]
r"""Type Alias for maybe wrapped values."""
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
