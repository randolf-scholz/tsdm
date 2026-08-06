r"""Tests for `tsdm.types.protocols.SupportsKwargs`."""

from collections.abc import Collection, Iterable
from typing import (
    Protocol,
    TypeIs,
    _ProtocolMeta as ProtocolMeta,
    runtime_checkable,
)

import pytest


@runtime_checkable
class SupportsKeysAndGetItem[K, V](Protocol):  # K, +V
    r"""Protocol for objects that support `__getitem__` and `keys`."""

    def keys(self) -> Collection[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


class _SupportsKwargsMeta(ProtocolMeta):
    r"""Metaclass for `SupportsKwargs`."""

    def __instancecheck__(cls, instance: object, /) -> TypeIs[SupportsKwargs]:  # noqa: N805
        return isinstance(instance, SupportsKeysAndGetItem) and all(
            isinstance(key, str)
            for key in instance.keys()  # noqa: SIM118
        )

    def __subclasscheck__(cls, subclass: type, /) -> TypeIs[type[SupportsKwargs]]:  # noqa: N805
        raise NotImplementedError("Cannot check whether a class is a SupportsKwargs.")


@runtime_checkable
class SupportsKwargs[V](Protocol, metaclass=_SupportsKwargsMeta):  # +V
    r"""Protocol for objects that support `**kwargs`."""

    def keys(self) -> Iterable[str]: ...
    def __getitem__(self, key: str, /) -> V: ...


def test_supports_kwargs() -> None:
    d_int = {0: 1, 1: 2}
    with pytest.raises(NotImplementedError):
        issubclass(d_int, SupportsKwargs)  # type: ignore
    assert isinstance(d_int, SupportsKwargs) is False  # pyrefly: ignore[unsafe-overlap]

    d_str = {"a": 1, "b": 2}
    with pytest.raises(NotImplementedError):
        issubclass(d_str, SupportsKwargs)  # type: ignore
    assert isinstance(d_str, SupportsKwargs) is True


def test_supportskwargs() -> None:
    r"""Test the SupportsKwargs protocol."""

    class StrKeys:
        r"""Dummy class that supports `**kwargs`."""

        @staticmethod
        def keys() -> list[str]:
            return ["some", "strings"]

        def __getitem__(self, key: str) -> int:
            return len(key)

    class IntKeys:
        r"""Dummy class that does not support `**kwargs`."""

        @staticmethod
        def keys() -> list[int]:
            return [1, 2]

        def __getitem__(self, key: int) -> int:
            return key

    assert isinstance(StrKeys(), SupportsKeysAndGetItem)
    assert isinstance(IntKeys(), SupportsKeysAndGetItem)
    assert isinstance(StrKeys(), SupportsKwargs)
    assert not isinstance(IntKeys(), SupportsKwargs)  # pyrefly: ignore[unsafe-overlap]
