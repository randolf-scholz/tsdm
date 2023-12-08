"""Utilities for testing and validation."""

__all__ = [
    "is_flattened",
    "is_dunder",
    "is_zipfile",
]

from collections.abc import Mapping
from pathlib import Path
from zipfile import BadZipFile, ZipFile


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
