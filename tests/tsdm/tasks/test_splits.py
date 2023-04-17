#!/usr/bin/env python
r"""Test converters to masked format etc."""

import logging

import pytest

logging.basicConfig(level=logging.INFO)
__logger__ = logging.getLogger(__name__)


@pytest.mark.skip(reason="not implemented")
def test_split_construction() -> None:
    """Test the construction of splits."""
    __logger__.info("Testing %s.", object)


def _main() -> None:
    test_split_construction()


if __name__ == "__main__":
    _main()
