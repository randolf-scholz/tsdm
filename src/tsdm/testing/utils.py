r"""Utilities for testing and validation."""

__all__ = [
    # functions
    "is_classvar",
    "is_dunder",
    "is_private",
    "is_zipfile",
]

from annotationlib import ForwardRef
from typing import ClassVar
from zipfile import BadZipFile, ZipFile

from typing_extensions import TypeForm

from tsdm.types import FilePath


def is_dunder(name: str, /) -> bool:
    r"""Check if the name is a dunder method.

    Equivalent to the regex pattern: ``^__.*__$``
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

    Equivalent to the regex pattern: ``^_.*[^_]$``
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


def is_classvar(tp: TypeForm, /) -> bool:
    r"""Check if the type annotation is a ClassVar."""
    if tp is ClassVar:
        return True
    if isinstance(tp, str | ForwardRef):
        # TODO: add support for ForwardRef
        raise NotImplementedError("ForwardRef / string annotation is not supported.")
    if (origin := getattr(tp, "__origin__", None)) is not None:
        return is_classvar(origin)
    return False
