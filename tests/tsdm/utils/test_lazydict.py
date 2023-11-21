r"""Test LazyDict."""

import logging
from collections.abc import MutableMapping

import pytest

from tsdm.utils import LazyDict, LazyValue

__logger__ = logging.getLogger(__name__)


@pytest.mark.filterwarnings("ignore:Using __ror__ with a non-LazyDict")
def test_lazydict() -> None:
    """Test the LazyDict class."""
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
        2: (no_input,),
        3: (single_input,),
        4: (positional_only, (1, 1, 1, 1)),
        5: (keyword_only, {"d": 1, "e": 1, "f": 1, "g": 1}),
        6: (generic, (1, 1, 1, 1), {"d": 1, "e": 1, "f": 1, "g": 1}),
    }
    ld = LazyDict(example_dict, fixed_value=3)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, LazyValue)

    for key in ld:
        assert isinstance(ld[key], int)

    EMPTY: LazyDict = LazyDict()

    # test __or__ operator with other LazyDict
    ld = EMPTY | LazyDict({0: lambda: 0})
    assert ld is not EMPTY, "__or__ should create a new dictionary"
    assert isinstance(ld, LazyDict), f"Got {type(ld)} instead of LazyDict."
    for value in ld.values():
        assert isinstance(value, LazyValue)

    # test __or__ operator with other dict
    ld = EMPTY | {0: lambda: 0}
    assert ld is not EMPTY, "__or__ should create a new dictionary"
    assert isinstance(ld, LazyDict), f"Got {type(ld)} instead of LazyDict."
    for value in ld.values():
        assert isinstance(value, LazyValue)

    # test __ror__ operatortsdm/utils/test_lazydict.py:73:
    empty: dict = {}
    other: dict = empty | LazyDict({0: lambda: 0})
    assert other is not empty, "__ror__ should create a new dictionary"
    assert isinstance(other, dict) and isinstance(other, LazyDict)
    # for value in other.values():
    #     assert isinstance(value, int)

    # test __ior__ operator
    ld = EMPTY
    ld |= {0: lambda: 0}
    assert ld is EMPTY, "__ior__ should modify existing dictionary"
    assert isinstance(ld, LazyDict), f"Got {type(ld)} instead of LazyDict."
    for value in ld.values():
        assert isinstance(value, LazyValue)


def test_lazydict_fromkeys() -> None:
    """Test the fromkeys method of LazyDict."""
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


def test_lazydict_copy() -> None:
    """Test the copy method of LazyDict."""
    LOGGER = __logger__.getChild(LazyDict.__name__)
    LOGGER.info("Testing %s", LazyDict.copy)

    ldA = LazyDict.fromkeys([1, 2, 3], 0)
    ldB = ldA.copy()
    assert isinstance(ldB, LazyDict)

    for (keyA, valueA), (keyB, valueB) in zip(ldA.items(), ldB.items()):
        assert keyA is keyB
        assert valueA is valueB
        assert isinstance(valueA, LazyValue)
        assert isinstance(valueB, LazyValue)

    # compute the value in the second dictionary
    for keyB in ldB:
        assert isinstance(ldB[keyB], int)

    # check that the first dictionary is still lazy
    for (keyA, valueA), (keyB, valueB) in zip(ldA.items(), ldB.items()):
        assert keyA is keyB
        assert valueA is not valueB
        assert isinstance(valueB, int)
        assert isinstance(valueA, LazyValue)
