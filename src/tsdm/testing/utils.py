r"""Utilities for testing and validation."""

__all__ = [
    # functions
    "is_builtin",
    "is_builtin_constant",
    "is_builtin_type",
    "is_dtype",
    "is_dunder",
    "is_na_value",
    "is_private",
    "is_scalar",
    "is_zipfile",
]

from inspect import isbuiltin
from types import EllipsisType, NoneType, NotImplementedType
from typing import Final, TypeGuard, TypeIs
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch
from pandas import NA, NaT

from tsdm.dtypes import DType
from tsdm.types.aliases import FilePath, PythonScalar

_BUILTIN_TYPES: Final[frozenset[type]] = frozenset(
    {
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        list,
        tuple,
        set,
        frozenset,
        dict,
        type,
        slice,
        range,
        object,
        NoneType,
        EllipsisType,
        NotImplementedType,
    }
)
r"""Builtin types https://docs.python.org/3/library/stdtypes.html."""


# NOTE: We ought to use TypeGuard here. see https://peps.python.org/pep-0742/#typeis-and-typeguard
def is_builtin_type(obj: object, /) -> TypeGuard[type]:
    r"""Check if the object is a builtin type."""
    try:
        return obj in _BUILTIN_TYPES
    except TypeError:
        return False


def is_builtin_constant(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    try:
        # XREF: Builtin constants https://docs.python.org/3/library/constants.html
        return obj in {None, True, False, Ellipsis, NotImplemented}
    except TypeError:
        return False


def is_builtin(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    return isbuiltin(obj) or is_builtin_constant(obj) or is_builtin_type(obj)


def is_dtype(arg: object, /) -> TypeIs[DType]:
    r"""Check if a string is a valid dtype."""
    return isinstance(
        arg,
        np.dtype
        | torch.dtype
        | pa.DataType
        | pd.api.extensions.ExtensionDtype
        | pl.DataType,
    ) or (isinstance(arg, type) and issubclass(arg, pl.DataType))


def is_na_value(obj: object, /) -> bool:
    r"""Check if the object is a NA value."""
    try:
        return bool((np.isscalar(obj) and np.isnan(obj)) or pd.isna(obj))
    except ValueError:
        return False


def is_scalar(obj: object, /) -> bool:
    r"""Check if an object is a basic type."""
    return (
        is_builtin_constant(obj)
        or isinstance(obj, PythonScalar.__value__)
        or np.isscalar(obj)
        or obj is NA
        or obj is NaT
    )


def is_dunder(name: str, /) -> bool:
    r"""Check if the name is a dunder method.

    Equivalent to the regex pattern: `^__.*__$`
    """
    return (
        name.isidentifier()
        and name.startswith("__")
        and not name.startswith("___")
        and name.endswith("__")
        and not name.endswith("___")
        and len(name) > 4
    )


def is_private(name: str, /) -> bool:
    r"""Check if the name is a private method.

    Equivalent to the regex pattern: `^_.*[^_]$`
    """
    return (
        name.isidentifier()
        and name.startswith("_")
        and not name.startswith("__")
        and len(name) > 1
    )


def is_zipfile(path: FilePath, /) -> bool:
    r"""Return `True` if the file is a zipfile."""
    try:
        with ZipFile(path):
            return True
    except BadZipFile, IsADirectoryError:
        return False
