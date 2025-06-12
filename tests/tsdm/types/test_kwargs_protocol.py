r"""Tests for `tsdm.types.protocols.SupportsKwargs`."""

import pytest

from tsdm.types.mixins import SupportsKeysAndGetItem
from tsdm.types.protocols import SupportsKwargs


def test_supports_kwargs() -> None:
    d_int = {0: 1, 1: 2}
    with pytest.raises(NotImplementedError):
        issubclass(d_int, SupportsKwargs)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportGeneralTypeIssues]
    assert isinstance(d_int, SupportsKwargs) is False

    d_str = {"a": 1, "b": 2}
    with pytest.raises(NotImplementedError):
        issubclass(d_str, SupportsKwargs)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType, reportGeneralTypeIssues]
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
    assert not isinstance(IntKeys(), SupportsKwargs)
