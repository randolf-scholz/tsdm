"""Utilities for testing and validation."""

__all__ = [
    "is_flattened",
    "is_dunder",
    "is_zipfile",
    "assert_protocol",
    "check_shared_attrs",
]

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from typing_extensions import get_protocol_members, is_protocol


def is_flattened(
    d: Mapping, /, *, key_type: type = object, val_type: type = Mapping
) -> bool:
    r"""Check if mapping is flattened."""
    return all(
        isinstance(key, key_type) and not isinstance(val, val_type)
        for key, val in d.items()
    )


def is_dunder(name: str, /) -> bool:
    r"""Check if the name is a dunder method."""
    return name.isidentifier() and name.startswith("__") and name.endswith("__")


def is_zipfile(path: Path, /) -> bool:
    r"""Return `True` if the file is a zipfile."""
    try:
        with ZipFile(path):
            return True
    except (BadZipFile, IsADirectoryError):
        return False


def assert_protocol(obj: Any, proto: type, /) -> None:
    """Assert that the object is a given protocol."""
    if not is_protocol(proto):
        raise TypeError(f"{proto} is not a protocol!")

    if isinstance(obj, type):
        match = issubclass(obj, proto)
        name = obj.__name__
    else:
        match = isinstance(obj, proto)
        name = obj.__class__.__name__

    if not match:
        raise AssertionError(
            f"{name} is not a {proto.__name__}!"
            f"\n Missing Attributes: {get_protocol_members(proto) - set(dir(obj))}"
        )


def check_shared_attrs(*classes: type, protocol: type) -> list[str]:
    """Check that all classes satisfy the protocol."""
    if not is_protocol(protocol):
        raise TypeError(f"{protocol} is not a protocol!")

    protocol_members = get_protocol_members(protocol)
    shared_members = set.intersection(*(set(dir(cls)) for cls in classes))
    missing_members = sorted(protocol_members - shared_members)
    superfluous_members = sorted(shared_members - protocol_members)

    if missing_members:
        raise AssertionError(f"Missing members: {missing_members}")

    return superfluous_members
