r"""Test LazyDict."""

import logging
from collections.abc import Callable, MutableMapping
from typing import Any, assert_type

import pytest

from tsdm.utils import LazyDict, LazyValue

__logger__ = logging.getLogger(__name__)

EMPTY_LAZYDICT: LazyDict = LazyDict()


def test_lazydict_init() -> None:
    r"""Test the LazyDict class."""
    LOGGER = __logger__.getChild(LazyDict.__name__)
    LOGGER.info("Testing.")

    def no_input():
        return 0

    def single_input(x):
        return x

    def positional_only(a, /, b, c=1, *args):
        return a + b + c + sum(args)

    def keyword_only(*, d, e=2, **kwargs):
        return d + e + sum(kwargs.values())

    def generic(a, /, b, c=1, *args, d, e=2, **kwargs):
        return a + b + c + sum(args) + d + e + sum(kwargs.values())

    example_dict = {
        0: no_input,
        1: single_input,
        # 2: (no_input,),
        # 3: (single_input,),
        4: (positional_only, (1, 1, 1, 1)),
        5: (keyword_only, {"d": 1, "e": 1, "f": 1, "g": 1}),
        6: (generic, (1, 1, 1, 1), {"d": 1, "e": 1, "f": 1, "g": 1}),
    }
    ld = LazyDict(example_dict)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, LazyValue)

    for key in ld:
        assert isinstance(ld[key], int)


def test_or() -> None:
    r"""Test `__or__` operator."""
    other = LazyDict({0: 0})
    ld = EMPTY_LAZYDICT | other
    assert ld is not EMPTY_LAZYDICT, "__or__ should create a new dictionary"
    assert isinstance(ld, LazyDict), f"Got {type(ld)} instead of LazyDict."
    for value in ld.values():
        assert isinstance(value, LazyValue)


def test_ror() -> None:
    r"""Test `__ror__` operator."""
    empty: dict = {}
    other = LazyDict({0: 0})
    with pytest.raises(NotImplementedError):
        _ = empty | other


def test_ior() -> None:
    r"""Test `__ior__` operator."""
    ld = EMPTY_LAZYDICT
    other = {0: 0}
    ld |= other
    assert ld is EMPTY_LAZYDICT, "__ior__ should modify existing dictionary"
    assert isinstance(ld, LazyDict), f"Got {type(ld)} instead of LazyDict."
    for value in ld.values():
        assert not isinstance(value, LazyValue)


def test_fromkeys() -> None:
    r"""Test the fromkeys method of LazyDict."""
    LOGGER = __logger__.getChild(LazyDict.__name__)
    LOGGER.info("Testing %s", LazyDict.fromkeys)

    ld = LazyDict.fromkeys([1, 2, 3], 0)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, LazyValue)

    for key in ld:
        assert isinstance(ld[key], int)


def test_copy() -> None:
    r"""Test the copy method of LazyDict."""
    LOGGER = __logger__.getChild(LazyDict.__name__)
    LOGGER.info("Testing %s", LazyDict.copy)

    ldA = LazyDict.fromkeys([1, 2, 3], 0)
    ldB = ldA.copy()
    assert isinstance(ldB, LazyDict)

    for (keyA, valueA), (keyB, valueB) in zip(ldA.items(), ldB.items(), strict=True):
        assert keyA is keyB
        assert valueA is valueB
        assert isinstance(valueA, LazyValue)
        assert isinstance(valueB, LazyValue)

    # compute the value in the second dictionary
    for keyB in ldB:
        assert isinstance(ldB[keyB], int)

    # check that the first dictionary is still lazy
    for (keyA, valueA), (keyB, valueB) in zip(ldA.items(), ldB.items(), strict=True):
        assert keyA is keyB
        assert valueA is not valueB
        assert isinstance(valueB, int)
        assert isinstance(valueA, LazyValue)


def test_init_type_inference() -> None:
    d1: dict[int, int] = {0: 0}
    ld1 = LazyDict(d1)
    assert_type(ld1, LazyDict[int, int])

    d2: dict[int, Callable[[], int]] = {0: lambda: 0}
    ld2 = LazyDict(d2)  # type: ignore[arg-type, var-annotated]
    assert_type(ld2, LazyDict[int, int])  # type: ignore[assert-type]

    # without type hints
    d3 = {0: lambda: 0, 1: lambda: 1, 2: lambda: 2}
    ld3 = LazyDict(d3)  # type: ignore[arg-type, var-annotated]
    assert_type(ld3, LazyDict[int, Any])
