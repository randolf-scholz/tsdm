r"""Utilities for testing and validation."""

__all__ = [
    # functions
    "assert_arrays_close",
    "assert_arrays_equal",
    "assert_protocol",
    "check_shared_interface",
    "is_builtin",
    "is_builtin_constant",
    "is_builtin_type",
    "is_dunder",
    "is_private",
    "is_flattened",
    "is_na_value",
    "is_scalar",
    "is_zipfile",
    "supports_issubclass",
]

from collections.abc import Iterable, Mapping, Sequence, Set as AbstractSet
from inspect import getmembers, isbuiltin, isdatadescriptor, ismethoddescriptor
from typing import Any, TypeGuard, TypeIs, get_protocol_members, is_protocol
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pl_testing
import pyarrow as pa
import torch
import torch.testing
from pandas import NA, NaT

from tsdm.constants import BUILTIN_CONSTANTS, BUILTIN_TYPES, NA_VALUES
from tsdm.dtypes import DType
from tsdm.types.aliases import FilePath, PythonScalar


def assert_arrays_equal[T: Any](array: T, reference: T, /) -> None:
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


def assert_arrays_close[T: Any](
    array: T,
    reference: T,
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
                array, reference, check_exact=False, abs_tol=atol, rel_tol=rtol
            )
        case pl.DataFrame():
            pl_testing.assert_frame_equal(
                array, reference, check_exact=False, abs_tol=atol, rel_tol=rtol
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


def get_descriptors_and_callables(cls: type, /) -> set[str]:
    r"""Return the descriptors and callables of a type."""
    return {
        name
        for name, attr in getmembers(cls)
        if (callable(attr) or ismethoddescriptor(attr) or isdatadescriptor(attr))
    }


# NOTE: We ought to use TypeGuard here. see https://peps.python.org/pep-0742/#typeis-and-typeguard
def is_builtin_type(obj: object, /) -> TypeGuard[type]:
    r"""Check if the object is a builtin type."""
    try:
        return obj in BUILTIN_TYPES
    except TypeError:
        return False


def is_dtype(arg: object, /) -> TypeIs[DType]:
    r"""Check if a string is a valid dtype."""
    return isinstance(
        arg, np.dtype | torch.dtype | pa.DataType | pd.api.extensions.ExtensionDtype
    )


def is_builtin_constant(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    try:
        return obj in BUILTIN_CONSTANTS
    except TypeError:
        return False


def is_builtin(obj: object, /) -> bool:
    r"""Check if the object is a builtin constant."""
    return isbuiltin(obj) or is_builtin_constant(obj) or is_builtin_type(obj)


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
        or isinstance(obj, PythonScalar.__value__)
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


def supports_issubclass(cls: type, /) -> bool:
    r"""Check if the class supports issubclass."""
    try:
        result = issubclass(cls, cls)
    except TypeError:
        return False
    if not result:
        raise AssertionError(f"{cls} is not a subclass of itself!")
    return True


_DEFAULT_EXCLUSIONS = frozenset(set(dir(object)) | {"__hash__"})
r"""Default excluded members for shared interface checks."""


def check_shared_interface(
    test_cases: Iterable[object],
    protocol: type,
    *,
    excluded_members: AbstractSet[str] = _DEFAULT_EXCLUSIONS,
    raise_on_extra: bool = True,
    raise_on_unsatisfied: bool = True,
) -> None:
    r"""Check that all classes satisfy the protocol."""
    proto_name = protocol.__name__
    if not is_protocol(protocol):
        raise TypeError(f"{protocol} is not a protocol!")

    interface = get_protocol_members(protocol)
    interfaces = {type(obj): set(dir(obj)) for obj in test_cases}

    shared_members = set.intersection(*interfaces.values())
    shared_members -= excluded_members - interface  # remove excluded members

    unsatisfied: dict[type, list[str]] = {
        name: missing
        for name, members in interfaces.items()
        if (missing := sorted(interface - members))
    }

    if unsatisfied:
        msg = (
            f"The following examples do not satisfy the protocol {proto_name!r}:"
            f"\n\t{unsatisfied}"
        )
        if raise_on_unsatisfied:
            raise AssertionError(msg)
        print(msg)

    if extra_members := sorted(shared_members - interface):
        msg = (
            f"Shared members not covered by protocol {proto_name!r}:\n\t{extra_members}"
        )
        if raise_on_extra:
            raise AssertionError(msg)
        print(msg)
