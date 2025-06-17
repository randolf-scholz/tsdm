r"""Tests for tsdm.

NOTE: We use `/tests/tsdm/...` layout to ensure that the tests are not imported.
"""

import pytest

from tests import pytest_xfail


def check_foo() -> None:
    raise AssertionError("known bug: lib/#1234")


def check_bar() -> None:
    raise AssertionError("known bug: lib/#4567")


def check_baz() -> None:
    raise RuntimeError("unknown bug")


@pytest.mark.xfail(raises=RuntimeError)
def test_xfail() -> None:
    r"""Test the pytest_xfail context manager."""
    with pytest_xfail("lib/#1234", strict=True, raise_on_exit=False) as ctx1:
        check_foo()

    with pytest_xfail("lib/#4567", strict=True, raise_on_exit=False) as ctx2:
        check_bar()

    check_baz()

    pytest_xfail.raise_if_any(ctx1, ctx2)
