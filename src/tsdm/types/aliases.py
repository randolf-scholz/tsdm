r"""Collection of Useful Type Aliases."""

__all__ = [
    # Custom Type Aliases
    "Axis",
    "Dims",
    "FilePath",
    "Kwargs",
    "Nested",
    "PandasObject",
    "Shape",
    "Size",
    "SplitID",
    "PandasNullable",
    # Dtype Aliases
    "NumpyDtype",
    "NumpyDtypeArg",
    "PandasDtype",
    "PandasDTypeArg",
    "DType",
    # Scalar Type Aliases
    "BuiltinScalar",
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
from typing import Any

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.api.typing import NAType
from pandas.core.dtypes.base import ExtensionDtype

from tsdm.types.protocols import SupportsKwargs

# region Custom Type Aliases -----------------------------------------------------------
type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Dims = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""  # FIXME: https://github.com/pytorch/pytorch/issues/64700
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type Shape = int | tuple[int, ...]
r"""Type Alias for shape-like objects (note: `ones(shape=None)` creates 0d-array."""
type Kwargs[T] = SupportsKwargs[T]
r"""Type Alias for keyword arguments."""
type Nested[T] = T | Collection["Nested[T]"] | Mapping[Any, "Nested[T]"]  # +T
r"""Type Alias for nested types (JSON-Like)."""
type FilePath = str | Path | os.PathLike[str]  # cf. pandas._typing.FilePath
r"""Type Alias for path-like objects."""
type SplitID = Hashable
r"""Type Alias for split identifiers."""
# endregion Custom Type Aliases --------------------------------------------------------

# region Dtype Aliases -----------------------------------------------------------------
type NumpyDtype = np.dtype
r"""Type Alias for `numpy` dtypes."""
type NumpyDtypeArg = str | type | NumpyDtype
r"""Type Alias for `numpy` dtype arguments."""
type PandasDtype = ExtensionDtype | NumpyDtype
r"""Type Alias for `pandas` dtype."""
type PandasDTypeArg = str | type | PandasDtype
r"""Type Alias for `pandas` dtype arguments."""
type DType = np.dtype | torch.dtype | type[ExtensionDtype]
r"""Type Alias for dtypes."""
type DTypeArg = str | type
r"""Type Alias for dtype arguments."""
type PandasObject = DataFrame | Series | Index | MultiIndex
r"""Type Alias for `pandas` objects."""
type PandasNullable[T] = T | NAType
r"""Type Alias for nullable types."""
# endregion Dtype Aliases --------------------------------------------------------------


# region Scalar Type Aliases -----------------------------------------------------------
type BuiltinScalar = bool | int | float | complex | str | bytes
r"""Type Alias for scalars."""
type StringScalar = str | bytes
r"""Type Alias for string scalars."""
type NumericalScalar = bool | int | float | complex
r"""Type Alias for numerical scalars."""
type TorchScalar = bool | int | float | str
r"""Type Alias for scalars allowed by torchscript."""
type TimeScalar = datetime | timedelta
r"""Type Alias for time scalars."""
type PythonScalar = bool | int | float | complex | str | bytes | datetime | timedelta
r"""Type Alias for Python scalars."""
# endregion Scalar Type Aliases --------------------------------------------------------


# region Maybe Type Aliases ------------------------------------------------------------
# NOTE: Maybe refers to types of the kind Union[T, Container[T]]
type MaybeList[T] = T | list[T]
r"""Type Alias for T | list[T]."""
type MaybeTuple[T] = T | tuple[T, ...]
r"""Type Alias for T | tuple[T, ...]."""
type MaybeFrozenset[T] = T | frozenset[T]
r"""Type Alias for T | frozenset[T]."""
type MaybeSet[T] = T | set[T]
r"""Type Alias for T | set[T]."""
type MaybeIterable[T] = T | Iterable[T]
r"""Type Alias for T | Iterable[T]."""
type MaybeCallable[T] = T | Callable[[], T]
r"""Type Alias for objects that maybe needs to be created first."""
type MaybeWrapped[T] = T | Callable[[], T] | Callable[[int], T]  # +T
r"""Type Alias for maybe wrapped values."""
# endregion Maybe Type Aliases ---------------------------------------------------------


# region Nested collections.abc --------------------------------------------------------
type NestedCollection[T] = Collection[T | "NestedCollection[T]"]
r"""Generic Type Alias for nested `Collection`."""
type NestedIterable[T] = Iterable[T | "NestedIterable[T]"]
r"""Generic Type Alias for nested `Iterable`."""
type NestedSequence[T] = Sequence[T | "NestedSequence[T]"]
r"""Generic Type Alias for nested `Sequence`."""
type NestedMapping[K, V] = Mapping[K, V | "NestedMapping[K, V]"]
r"""Generic Type Alias for nested `Mapping`."""
type NestedMutableMapping[K, V] = MutableMapping[K, V | "NestedMutableMapping[K, V]"]
r"""Generic Type Alias for nested `MutableMapping`."""
# endregion Nested collections.abc -----------------------------------------------------


# region Nested Builtins ---------------------------------------------------------------
type NestedDict[K, V] = dict[K, V | "NestedDict[K, V]"]
r"""Generic Type Alias for nested `dict`."""
type NestedList[T] = list[T | "NestedList[T]"]
r"""GenericType Alias for nested `list`."""
type NestedSet[T] = set[T | "NestedSet[T]"]
r"""Generic Type Alias for nested `set`."""
type NestedFrozenSet[T] = frozenset[T | "NestedFrozenSet[T]"]
r"""Generic Type Alias for nested `set`."""
type NestedTuple[T] = tuple[T | "NestedTuple[T]", ...]
r"""Generic Type Alias for nested `tuple`."""
type NestedBuiltin[T] = (
    T
    | tuple[T, ...]
    | tuple["NestedBuiltin[T]", ...]
    | tuple[T | "NestedBuiltin[T]", ...]
    | set[T]
    | set["NestedBuiltin[T]"]
    | set[T | "NestedBuiltin[T]"]
    | frozenset[T]
    | frozenset["NestedBuiltin[T]"]
    | frozenset[T | "NestedBuiltin[T]"]
    | list[T]
    | list["NestedBuiltin[T]"]
    | list[T | "NestedBuiltin[T]"]
    | dict[str, T]
    | dict[str, "NestedBuiltin[T]"]
    | dict[str, T | "NestedBuiltin[T]"]
)
r"""Type Alias for nested builtins."""
# endregion Nested Builtins ------------------------------------------------------------


# region Nested Configuration ----------------------------------------------------------
type JSON_LeafType = None | bool | int | float | str
type TOML_LeafType = None | bool | int | float | str | datetime
type YAML_LeafType = None | bool | int | float | str | datetime

type JSON = JSON_LeafType | list["JSON"] | dict[str, "JSON"]
r"""Type Alias for JSON-Like objects."""
type TOML = TOML_LeafType | list["TOML"] | dict[str, "TOML"]
r"""Type Alias for JSON-Like objects."""
type YAML = YAML_LeafType | list["YAML"] | dict[str, "YAML"]
r"""Type Alias for JSON-Like objects."""
# endregion Nested Configuration -------------------------------------------------------
