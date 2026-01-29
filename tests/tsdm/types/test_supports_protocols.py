r"""Tests for SupportsX type protocols."""

from collections.abc import Mapping
from typing import assert_type

from tsdm.types._protocols import SupportsKeysAndGetItem


def test_supportskeysgetitem() -> None:
    r"""Test the SupportsKeysAndGetItem protocol."""

    def foo[K, V](x: Mapping[K, V]) -> SupportsKeysAndGetItem[K, V]:
        return x

    assert_type(foo({"a": 1}), SupportsKeysAndGetItem[str, int])
