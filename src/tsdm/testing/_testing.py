r"""Utilities for testing and validation."""

__all__ = [
    # Functions
    "assert_arrays_close",
    "assert_arrays_equal",
    "assert_protocol",
    "check_shared_attrs",
    "is_builtin",
    "is_builtin_constant",
    "is_builtin_function",
    "is_builtin_type",
    "is_dunder",
    "is_flattened",
    "is_na_value",
    "is_scalar",
    "is_zipfile",
]

from collections.abc import Mapping, Sequence
from inspect import isbuiltin as is_builtin_function
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing
import torch
import torch.testing
from pandas import NA, NaT
from typing_extensions import Any, get_protocol_members, is_protocol

from tsdm.constants import BUILTIN_CONSTANTS, BUILTIN_TYPES, NA_VALUES
from tsdm.types.aliases import PythonScalar
from tsdm.types.variables import T_contra


def assert_arrays_equal(array: T_contra, reference: T_contra, /) -> None:
    r"""Assert that the arrays are equal."""
    if type(array) is not type(reference):
        raise AssertionError(f"{type(array)=} != {type(reference)=}")

    match array:
        case pd.Series():
            pd.testing.assert_series_equal(array, reference)
        case pd.Index():
            pd.testing.assert_index_equal(array, reference)
        case pd.DataFrame():
            pd.testing.assert_frame_equal(array, reference)
        case np.ndarray():
            np.testing.assert_array_equal(array, reference)
        case pl.Series():
            pl_testing.assert_series_equal(array, reference)
        case pl.DataFrame():
            pl_testing.assert_frame_equal(array, reference)
        case torch.Tensor():
            torch.testing.assert_close(array, reference, rtol=0, atol=0)
        case Sequence() as seq:
            if len(array) != len(reference):
                raise AssertionError(f"{len(array)=} != {len(reference)=}")
            if any(a != b for a, b in zip(seq, reference, strict=True)):
                raise AssertionError(f"{array=} != {reference=}")
        case _:
            raise TypeError(f"Unsupported {type(array)=}")


def assert_arrays_close(
    array: T_contra,
    reference: T_contra,
    /,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    r"""Assert that the arrays are close within tolerance."""
    if type(array) is not type(reference):
        raise AssertionError(f"{type(array)=} != {type(reference)=}")

    match array:
        case pd.Series():
            pd.testing.assert_series_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case pd.Index():
            pd.testing.assert_index_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case pd.DataFrame():
            pd.testing.assert_frame_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case np.ndarray():
            np.testing.assert_allclose(array, reference, atol=atol, rtol=rtol)
        case pl.Series():
            pl_testing.assert_series_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case pl.DataFrame():
            pl_testing.assert_frame_equal(
                array, reference, check_exact=False, atol=atol, rtol=rtol
            )
        case torch.Tensor() as tensor:
            torch.testing.assert_close(tensor, reference, atol=atol, rtol=rtol)
        case Sequence() as seq:
            if len(array) != len(reference):
                raise AssertionError(f"{len(array)=} != {len(reference)=}")
            if any(
                abs(a - b) > atol + rtol * abs(b)
                for a, b in zip(seq, reference, strict=True)
            ):
                raise AssertionError(f"{array=} != {reference=}")
        case _:
            raise TypeError(f"Unsupported {type(array)=}")


def is_builtin_type(obj: object, /) -> bool:
    r"""Check if the object is a builtin type."""
    try:
        return obj in BUILTIN_TYPES
    except TypeError:
        return False


def is_builtin_constant(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    try:
        return obj in BUILTIN_CONSTANTS
    except TypeError:
        return False


def is_builtin(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    return is_builtin_function(obj) or is_builtin_constant(obj) or is_builtin_type(obj)


def is_na_value(obj: object, /) -> bool:
    r"""Check if the object is a NA value."""
    try:
        return obj in NA_VALUES
    except TypeError:
        return False


def is_scalar(obj: object, /) -> bool:
    r"""Check if an object is a basic type."""
    return (
        is_builtin_constant(obj)
        or isinstance(obj, PythonScalar)  # type: ignore[misc,arg-type]
        or obj is NA
        or obj is NaT
    )


def is_flattened(
    d: Mapping, /, *, key_type: type = object, val_type: type = Mapping
) -> bool:
    r"""Check if mapping is flattened."""
    return all(
        isinstance(key, key_type) and not isinstance(val, val_type)
        for key, val in d.items()
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


def is_zipfile(path: Path, /) -> bool:
    r"""Return `True` if the file is a zipfile."""
    try:
        with ZipFile(path):
            return True
    except (BadZipFile, IsADirectoryError):
        return False


def assert_protocol(obj: Any, proto: type, /, *, expected: bool = True) -> None:
    r"""Assert that the object is a given protocol."""
    if not is_protocol(proto):
        raise TypeError(f"{proto} is not a protocol!")

    if isinstance(obj, type):
        match = issubclass(obj, proto)
        name = obj.__name__
    else:
        match = isinstance(obj, proto)
        name = obj.__class__.__name__

    member = "a subtype" if isinstance(obj, type) else "an instance"
    msg = f"{name!r} is {'not' if expected else ''} {member} of {proto.__name__!r}!"
    missing_attrs = sorted(get_protocol_members(proto) - set(dir(obj)))

    if expected and not match:
        raise AssertionError(f"{msg}\n Missing Attributes: {missing_attrs}")
    if match and not expected:
        raise AssertionError(msg)


def check_shared_attrs(*classes: type, protocol: type) -> list[str]:
    r"""Check that all classes satisfy the protocol."""
    if not is_protocol(protocol):
        raise TypeError(f"{protocol} is not a protocol!")

    protocol_members = get_protocol_members(protocol)
    shared_members = set.intersection(*(set(dir(cls)) for cls in classes))
    missing_members = sorted(protocol_members - shared_members)
    superfluous_members = sorted(shared_members - protocol_members)

    if missing_members:
        raise AssertionError(f"Missing members: {missing_members}")

    return superfluous_members
