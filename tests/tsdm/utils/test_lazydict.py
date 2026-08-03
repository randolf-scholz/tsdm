r"""Test LazyDict."""
# mypy: disable-error-code="no-untyped-def"

import logging
from collections.abc import Callable, Mapping, MutableMapping
from typing import Final, assert_type

import pytest

from tsdm.utils.lazydict import LazyDict, LazyValue, lazy_dict

__logger__ = logging.getLogger(__name__)


EMPTY_LAZYDICT: Final[LazyDict] = LazyDict()
EMPTY_DICT: Final[dict] = {}


def lazy_int() -> int:
    return 42


def lazy_float() -> float:
    return 1.23


def lazy_str() -> str:
    return "hello"


def test_lazy_dict_function() -> None:
    d0 = lazy_dict()
    assert_type(d0, LazyDict)
    d1 = lazy_dict({})
    assert_type(d1, LazyDict)
    d2 = lazy_dict(pi=lambda: 1.23)
    assert_type(d2, LazyDict[str, float])
    d3 = lazy_dict({"pi": lambda: 1.23})
    assert_type(d3, LazyDict[str, float])
    d4 = lazy_dict({0: lambda: 0.0, 1: lambda: 1.0})
    assert_type(d4, LazyDict[int, float])


def test_init_type_inference() -> None:
    d1 = {0: lazy_int, 1: lazy_int, 2: lazy_int}
    ld1 = lazy_dict(d1)
    assert_type(ld1, LazyDict[int, int])

    d2: dict[int, Callable[[], int]] = {0: lazy_int, 1: lazy_int, 2: lazy_int}
    ld2 = lazy_dict(d2)
    assert_type(ld2, LazyDict[int, int])

    # without type hints
    # FIXME: https://github.com/microsoft/pyright/issues/8638
    d3 = {0: lazy_int, 1: lazy_int, 2: lazy_int}
    ld3 = lazy_dict(d3)
    assert_type(ld3, LazyDict[int, int])
    assert all(isinstance(value, LazyValue) for value in ld3.values())
    assert isinstance(ld3[0], int)


def test_lazy_dict_init() -> None:
    # check unbound initializers
    d1 = LazyDict()
    assert_type(d1, LazyDict)
    d2 = LazyDict({"x": 0.0})
    assert_type(d2, LazyDict[str, float])

    # check bound initializers
    d3 = LazyDict[str, float]()
    assert_type(d3, LazyDict[str, float])
    d4 = LazyDict[str, float]({})
    assert_type(d4, LazyDict[str, float])
    d5 = LazyDict[str, float]({"x": 0.0})
    assert_type(d5, LazyDict[str, float])


def test_lazy_dict_new() -> None:
    # no arguments
    no0 = LazyDict.new()
    no1 = LazyDict[int, float].new()
    assert_type(no0, LazyDict)
    assert_type(no1, LazyDict[int, float])  # type: ignore[assert-type]

    # positional arguments
    po0 = LazyDict.new({1: lazy_float})
    po1 = LazyDict[int, float].new({1: lazy_float})
    assert_type(po0, LazyDict[int, float])
    assert_type(po1, LazyDict[int, float])

    # keyword arguments
    kw0 = LazyDict.new(foo=lazy_float)
    kw1 = LazyDict[str, float].new(foo=lazy_float)
    assert_type(kw0, LazyDict[str, float])
    assert_type(kw1, LazyDict[str, float])

    # mixed key types
    mix_k0 = LazyDict.new({1: lazy_float}, foo=lazy_float)
    mix_k1 = LazyDict[int | str, float].new({1: lazy_float}, foo=lazy_float)
    assert_type(mix_k0, LazyDict[int | str, float])
    assert_type(mix_k1, LazyDict[int | str, float])

    # mixed value types
    mix_v0 = LazyDict.new({"x": lazy_float}, foo=lazy_str)
    mix_v1 = LazyDict[str, str | float].new({"x": lazy_float}, foo=lazy_str)
    assert_type(mix_v0, LazyDict[str, str | float])
    assert_type(mix_v1, LazyDict[str, str | float])

    # mixed key and value types
    mix_kv0 = LazyDict.new({"x": lazy_float, 1: lazy_str}, foo=lazy_float)
    mix_kv1 = LazyDict[str | int, str | float].new(
        {"x": lazy_float, 1: lazy_str}, foo=lazy_float
    )
    assert_type(mix_kv0, LazyDict[str | int, str | float])
    assert_type(mix_kv1, LazyDict[str | int, str | float])


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
        0: LazyValue(no_input, (), {}),
        1: LazyValue(single_input, (1,), {}),
        # 2: (no_input,),
        # 3: (single_input,),
        4: LazyValue(positional_only, (1, 1, 1, 1), {}),
        5: LazyValue(keyword_only, (), {"d": 1, "e": 1, "f": 1, "g": 1}),
        6: LazyValue(generic, (1, 1, 1, 1), {"d": 1, "e": 1, "f": 1, "g": 1}),
    }
    ld = lazy_dict(example_dict)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, LazyValue)

    for key in ld:
        assert isinstance(ld[key], int)


@pytest.mark.parametrize("other", [EMPTY_DICT, EMPTY_LAZYDICT])
def test_or(other: Mapping) -> None:
    r"""Test `__or__` operator."""
    self = lazy_dict({"x": lazy_float})
    assert_type(self, LazyDict[str, float])

    ld = self | other
    assert ld is not other, "__or__ should create a new dictionary"
    assert ld is not self, "__or__ should create a new dictionary"
    assert isinstance(ld, LazyDict), f"Got {type(ld)} instead of LazyDict."
    assert ld == self

    for value in ld.values():
        assert isinstance(value, LazyValue)


@pytest.mark.parametrize("other", [EMPTY_DICT, EMPTY_LAZYDICT])
def test_ror(other: Mapping) -> None:
    r"""Test `__ror__` operator."""
    self = lazy_dict({"x": lazy_float})
    assert_type(self, LazyDict[str, float])

    ld = other | self
    assert ld is not other, "__or__ should create a new dictionary"
    assert ld is not self, "__or__ should create a new dictionary"
    assert isinstance(ld, LazyDict), f"Got {type(ld)} instead of LazyDict."
    assert ld == self

    for value in ld.values():
        assert isinstance(value, LazyValue)


@pytest.mark.parametrize("other", [EMPTY_DICT, EMPTY_LAZYDICT])
def test_ior(other: Mapping) -> None:
    r"""Test `__ior__` operator."""
    self = lazy_dict({"x": lazy_float})
    assert_type(self, LazyDict[str, float])
    self |= other

    for value in self.values():
        assert isinstance(value, LazyValue)


def test_fromkeys() -> None:
    r"""Test the `fromkeys` method of `LazyDict`."""
    LOGGER = __logger__.getChild(LazyDict.__name__)
    LOGGER.info("Testing %s", LazyDict.fromkeys)

    ld = LazyDict.fromkeys([1, 2, 3], 0)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, int)

    for key in ld:
        assert isinstance(ld[key], int)


def test_get() -> None:
    r"""Test the `get` method of `LazyDict`."""
    # get should return non-lazy values
    ld = LazyDict.from_func([1, 2, 3], lambda _: 0)
    assert ld.get(1) == 0


def test_get_lazy() -> None:
    r"""Test the `get` method of `LazyDict` with lazy values."""
    # get should return lazy values
    ld = LazyDict.from_func([1, 2, 3], lambda x: x**2)

    assert isinstance(ld.get_lazy(2), LazyValue)
    assert ld.get_lazy(5, "foo") == "foo"


def test_from_func() -> None:
    r"""Test the `from_func` method of `LazyDict`."""
    LOGGER = __logger__.getChild(LazyDict.__name__)
    LOGGER.info("Testing %s", LazyDict.fromkeys)

    ld = LazyDict.from_func([1, 2, 3], lambda x: x**2)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, LazyValue)

    for key in ld:
        assert isinstance(ld[key], int)
        assert ld[key] == key**2


def test_copy() -> None:
    r"""Test the copy method of LazyDict."""
    LOGGER = __logger__.getChild(LazyDict.__name__)
    LOGGER.info("Testing %s", LazyDict.copy)

    ldA = LazyDict.fromkeys([1, 2, 3], LazyValue(lazy_int))
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
