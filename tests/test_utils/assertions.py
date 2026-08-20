__all__ = [
    "assert_arrays_close",
    "assert_arrays_equal",
    "assert_protocol",
    "check_shared_interface",
    "supports_issubclass",
]

from collections.abc import Iterable, Sequence, Set as AbstractSet
from typing import Any, get_protocol_members, is_protocol

import numpy as np
import pandas as pd
import polars as pl
import torch
from polars import testing as pl_testing


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


def supports_issubclass(cls: type, /) -> bool:
    r"""Check if the class supports issubclass."""
    try:
        result = issubclass(cls, cls)
    except TypeError:
        return False
    if not result:
        raise AssertionError(f"{cls} is not a subclass of itself!")
    return True


def assert_protocol(obj: object, proto: type, /) -> None:
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
    msg = f"{name!r} is not a {member} of {proto.__name__!r}!"
    missing_attrs = sorted(get_protocol_members(proto) - set(dir(obj)))

    if not match:
        raise AssertionError(f"{msg}\n Missing Attributes: {missing_attrs}")
