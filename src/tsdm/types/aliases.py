r"""Collection of Useful Type Aliases."""

__all__ = [
    # Type Qualifiers
    "Fittable",
    "Derivable",
    "DerivedField",
    "FittedField",
    # Generic Type Aliases
    "DictArg",
    "IndexArg",
    "Indexer",
    "Label",
    "LabelArg",
    "MultiIndexer",
    # Custom Type Aliases
    "Axis",
    "Dims",
    "DirPath",
    "FilePath",
    "PathLike",
    "Shape",
    "Size",
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
    "Nested",
    "NestedMapping",
    "NestedDict",
    "NestedBuiltin",
    # Fields
    "TS",
    "TS_meta",
    "SC",
    "SC_meta",
    "CS",
    "CS_meta",
    "TS_FIELDS",
    "TSC_FIELDS",
]


import os
from collections.abc import (
    Collection,
    Iterable,
    Mapping,
)
from datetime import datetime, timedelta
from types import EllipsisType
from typing import Annotated, Literal

# region field types -------------------------------------------------------------------
type TS = Literal["timeseries"]
type TS_meta = Literal["timeseries_metadata"]
type SC = Literal["static_covariates"]
type SC_meta = Literal["static_covariates_metadata"]
type CS = Literal["constants"]
type CS_meta = Literal["constants_metadata"]
type TS_FIELDS = TS | TS_meta | SC | SC_meta
type TSC_FIELDS = TS | TS_meta | SC | SC_meta | CS | CS_meta
# endregion ----------------------------------------------------------------------------

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

# region custom type aliases -----------------------------------------------------------
type Axis = None | int | tuple[int, ...]
r"""Type Alias for axestype ."""
type Dims = None | int | list[int]
r"""Type Alias for dimensions compatible with torchscript."""  # FIXME: https://github.com/pytorch/pytorch/issues/64700
type Size = int | tuple[int, ...]
r"""Type Alias for size-like objects (note: `sample(size=None)` creates scalar."""
type Shape = int | tuple[int, ...]
r"""Type Alias for shape-like objects (note: `ones(shape=None)` creates 0d-array."""
type PathLike = str | os.PathLike[str]
r"""Type Alias for path-like objects."""
type FilePath = str | os.PathLike[str]
r"""Type Alias for path-like objects pointing to file."""
type DirPath = str | os.PathLike[str]
r"""Type Alias for path-like objects pointing to directory."""
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

# region generic type aliases ----------------------------------------------------------
type DictArg[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
r"""Type Alias for dictionary-like arguments."""
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
