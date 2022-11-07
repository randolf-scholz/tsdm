#!/usr/bin/env python
r"""Test LazyDict."""

import logging
from collections.abc import MutableMapping

from tsdm.utils.lazydict import LazyDict, LazyFunction

__logger__ = logging.getLogger(__name__)


def test_lazydict():
    r"""Test LazyDict."""

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

    d = {
        0: no_input,
        1: single_input,
        2: (no_input,),
        3: (single_input,),
        4: (positional_only, (1, 1, 1, 1)),
        5: (keyword_only, {"d": 1, "e": 1, "f": 1, "g": 1}),
        6: (generic, (1, 1, 1, 1), {"d": 1, "e": 1, "f": 1, "g": 1}),
    }
    ld = LazyDict(d, answer=42)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, LazyFunction)

    for key in ld:
        assert isinstance(ld[key], int)


def test_fromkeys():
    r"""Test if fromkeys works."""
    ld = LazyDict.fromkeys([1, 2, 3], 0)

    assert isinstance(ld, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld.values():
        assert isinstance(value, LazyFunction)

    for key in ld:
        assert isinstance(ld[key], int)


def test_copy():
    r"""Test if copying works."""
    ld = LazyDict.fromkeys([1, 2, 3], 0)
    ld2 = ld.copy()

    assert isinstance(ld2, LazyDict)
    assert isinstance(ld, dict)
    assert isinstance(ld, MutableMapping)

    for value in ld2.values():
        assert isinstance(value, LazyFunction)

    for key in ld2:
        assert isinstance(ld2[key], int)


def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    __logger__.info("Testing lazydict ...")
    test_lazydict()
    __logger__.info("Done.")


if __name__ == "__main__":
    _main()
