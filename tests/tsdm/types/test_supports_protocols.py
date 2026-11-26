r"""Tests for SupportsX type protocols."""

from collections.abc import Mapping
from typing import Protocol, assert_type, get_protocol_members, runtime_checkable

from tsdm.types import SupportsKeysAndGetItem, protocol_exclude


def test_supportskeysgetitem() -> None:
    r"""Test the SupportsKeysAndGetItem protocol."""

    def foo[K, V](x: Mapping[K, V]) -> SupportsKeysAndGetItem[K, V]:
        return x

    assert_type(foo({"a": 1}), SupportsKeysAndGetItem[str, int])


def test_protocol_exclude_inherits_from_each_base() -> None:
    r"""A composed protocol keeps both exclusions and its structural members."""

    class NonString(Protocol, metaclass=protocol_exclude, excluded=(str,)):
        def __len__(self) -> int: ...

    class NonMapping(Protocol, metaclass=protocol_exclude, excluded=(Mapping,)):
        pass

    @runtime_checkable
    class SizedNonMapping(NonString, NonMapping, Protocol):
        pass

    assert get_protocol_members(SizedNonMapping) == {"__len__"}
    assert isinstance([1], SizedNonMapping)
    assert not isinstance("abc", SizedNonMapping)
    assert not issubclass(dict, SizedNonMapping)
