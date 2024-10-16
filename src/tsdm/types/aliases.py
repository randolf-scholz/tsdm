r"""Collection of Useful Type Aliases."""

__all__ = [
    # Type Qualifiers
    "Fittable",
    "Derivable",
    "DerivedField",
    "FittedField",
    # Generic Type Aliases
    "MaybeNA",
    "Nested",
    "Thunk",
    # Custom Type Aliases
    "Axis",
    "Dims",
    "FilePath",
    "Indexer",
    "Label",
    "MultiIndexer",
    "Shape",
    "Size",
    # Dtype Aliases
    "NumpyDtype",
    "NumpyDtypeArg",
    "PandasDtype",
    "PandasDTypeArg",
    "PandasObject",
    "DType",
    # Scalar Type Aliases
    "BuiltinScalar",
    "StringScalar",
    "NumericalScalar",
    "TorchScalar",
    "TimeScalar",
    "PythonScalar",
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
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)
from datetime import datetime, timedelta
from pathlib import Path
from types import EllipsisType
from typing import Annotated, Any

import numpy as np
import torch
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.api.typing import NAType
from pandas.core.dtypes.base import ExtensionDtype

# region type qualifiers ---------------------------------------------------------------
type Fittable[T] = Annotated[T, "Fittable"]
r"""Type Alias for fields that can be fitted."""
type Derivable[T] = Annotated[T, "Derivable"]
r"""Type Alias for fields that can be derived automatically."""
type DerivedField[T] = Annotated[T, "DerivedField"]
r"""Type Alias for fields that are derived automatically."""
type FittedField[T] = Annotated[T, "FittedField"]
r"""Type Alias for fields that are fitted automatically."""
# endregion type qualifiers ------------------------------------------------------------


# region generic type aliases ----------------------------------------------------------
type Thunk[T] = Callable[[], T]
r"""Type Alias for lazy evaluation."""
type MaybeNA[T] = T | NAType
r"""Type Alias for nullable types."""
type Nested[T] = T | Collection["Nested[T]"] | Mapping[Any, "Nested[T]"]  # +T
r"""Type Alias for nested types (JSON-Like)."""
# endregion generic type aliases -------------------------------------------------------


# region custom type aliases -----------------------------------------------------------
type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Dims = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""  # FIXME: https://github.com/pytorch/pytorch/issues/64700
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type Shape = int | tuple[int, ...]
r"""Type Alias for shape-like objects (note: `ones(shape=None)` creates 0d-array."""
type FilePath = str | Path | os.PathLike[str]  # cf. pandas._typing.FilePath
r"""Type Alias for path-like objects."""
# endregion custom type aliases --------------------------------------------------------


# region aliases for indexing ----------------------------------------------------------
type IndexArg = None | int | slice | range | list[int] | list[bool] | EllipsisType
r"""Type alias for `__getitem__` argument for tensors."""
type MultiIndexer = IndexArg | tuple[IndexArg, ...]
r"""Indexer that always returns a sub-tensor."""
type Indexer = int | tuple[int, ...] | MultiIndexer
r"""Type hint for `__getitem__` argument for tensors."""
type LabelArg = None | int | str | slice | range | list[int] | list[str] | EllipsisType
r"""Type Alias for `__getitem__` argument for tabular objects."""
type Label = LabelArg | tuple[LabelArg, ...]
r"""Type Alias for `__getitem__` argument for tabular objects."""
# endregion aliases for indexing -------------------------------------------------------


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
    | tuple[T, ...]  # leaf-tuple
    | tuple["NestedBuiltin[T]", ...]
    | set[T]  # leaf-set
    | set["NestedBuiltin[T]"]
    | frozenset[T]  # leaf-frozenset
    | frozenset["NestedBuiltin[T]"]
    | list[T]  # leaf-list
    | list["NestedBuiltin[T]"]
    | dict[str, T]  # leaf-dict
    | dict[str, "NestedBuiltin[T]"]
)
r"""Type Alias for nested builtins."""
# endregion Nested Builtins ------------------------------------------------------------


# region Nested Configuration ----------------------------------------------------------
type JSON_LeafType = None | bool | int | float | str
type TOML_LeafType = None | bool | int | float | str | datetime
type YAML_LeafType = None | bool | int | float | str | datetime

type JSON = JSON_LeafType | list["JSON"] | dict[str, "JSON"]
r"""Type Alias for JSON-Like objects."""
type YAML = YAML_LeafType | list["YAML"] | dict[str, "YAML"]
r"""Type Alias for JSON-Like objects."""
type TOML = TOML_LeafType | list["TOML"] | dict[str, "TOML"]
r"""Type Alias for JSON-Like objects."""
# endregion Nested Configuration -------------------------------------------------------
