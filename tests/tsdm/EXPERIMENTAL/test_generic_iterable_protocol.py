r"""Tests for GenericIterable protocol."""

from collections.abc import Iterator
from types import GenericAlias
from typing import Protocol, runtime_checkable


@runtime_checkable
class GenericIterable[T](Protocol):  # +T
    r"""Does not work currently!"""

    # FIXME: https://github.com/python/cpython/issues/112319
    def __class_getitem__(cls, item: type, /) -> GenericAlias: ...
    def __iter__(self) -> Iterator[T]: ...
