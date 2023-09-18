#!/usr/bin/env python
r"""TODO: Add Test summary."""

import logging

from tsdm.utils.decorators import IterItems, IterKeys, IterValues

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


def test_iteritems() -> None:
    r"""Test `IterItems`."""
    mapping_obj = {"a": 1, "b": 2, "c": 3}
    iter_obj = IterItems(mapping_obj)

    # test __getitem__
    for key in mapping_obj.keys():
        assert iter_obj[key] == mapping_obj[key]

    # test .get()
    for key in mapping_obj.keys():
        assert iter_obj.get(key) == mapping_obj[key]

    # test .keys()
    for key in iter_obj.keys():
        assert isinstance(key, str)

    # test .values()
    for value in iter_obj.values():
        assert isinstance(value, int)

    # test .items()
    for item in iter_obj.items():
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], int)

    # test __iter__
    for element in iter_obj:
        assert isinstance(element, tuple)
        assert len(element) == 2
        assert isinstance(element[0], str)
        assert isinstance(element[1], int)


def test_iterkeys() -> None:
    r"""Test `IterItems`."""
    mapping_obj = {"a": 1, "b": 2, "c": 3}
    iter_obj = IterKeys(mapping_obj)

    # test __getitem__
    for key in mapping_obj.keys():
        assert iter_obj[key] == mapping_obj[key]

    # test .get()
    for key in mapping_obj.keys():
        assert iter_obj.get(key) == mapping_obj[key]

    # test .keys()
    for key in iter_obj.keys():
        assert isinstance(key, str)

    # test .values()
    for value in iter_obj.values():
        assert isinstance(value, int)

    # test .items()
    for item in iter_obj.items():
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], int)

    # test __iter__
    for element in iter_obj:
        assert isinstance(element, str)


def test_itervalues() -> None:
    r"""Test `IterItems`."""
    mapping_obj = {"a": 1, "b": 2, "c": 3}
    iter_obj = IterValues(mapping_obj)

    # test __getitem__
    for key in mapping_obj.keys():
        assert iter_obj[key] == mapping_obj[key]

    # test .get()
    for key in mapping_obj.keys():
        assert iter_obj.get(key) == mapping_obj[key]

    # test .keys()
    for key in iter_obj.keys():
        assert isinstance(key, str)

    # test .values()
    for value in iter_obj.values():
        assert isinstance(value, int)

    # test .items()
    for item in iter_obj.items():
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], int)

    # test __iter__
    for element in iter_obj:
        assert isinstance(element, int)


def _main() -> None:
    test_iteritems()
    test_iterkeys()
    test_itervalues()


if __name__ == "__main__":
    _main()
