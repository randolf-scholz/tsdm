r"""Tests for tsdm.

NOTE: We use `/tests/tsdm/...` layout to ensure that the tests are not imported.
"""

import pytest

from test_utils import pytest_xfail

# pytest does not publicly expose the XFailed exception type, so we capture it here.
try:
    pytest.xfail()
except BaseException as exc:
    XFailed = type(exc)
    if XFailed.__name__ != "XFailed":
        raise
else:
    raise RuntimeError("Could not determine pytest XFailed exception type.")


def raises_assertionerror() -> None:
    raise AssertionError("known bug: lib/#1234")


def raises_valueerror() -> None:
    raise ValueError("known bug: lib/#4567")


def raises_runtimeerror() -> None:
    raise RuntimeError("unknown bug")


def test_xfail_raises_success() -> None:
    with (
        pytest.raises(XFailed),
        pytest_xfail("lib/#1234", raises=AssertionError, strict=True),
    ):
        raises_assertionerror()


def test_xfail_raises_failed() -> None:
    with (
        pytest.raises(AssertionError),
        pytest_xfail("lib/#1234", raises=ValueError, strict=True),
    ):
        raises_assertionerror()


def test_xfail_late_capture() -> None:
    with pytest_xfail("lib/#1234", raises=AssertionError, defer_xfail=True) as check1:
        raises_assertionerror()

    with pytest_xfail("lib/#4567", raises=ValueError, defer_xfail=True) as check2:
        raises_valueerror()

    with pytest.raises(XFailed):
        pytest_xfail.any_failed(check1, check2)


def test_xfail_late_no_failure() -> None:
    with pytest_xfail(
        "lib/#1234", raises=AssertionError, strict=False, defer_xfail=True
    ) as check1:
        pass

    with pytest_xfail(
        "lib/#4567", raises=ValueError, strict=False, defer_xfail=True
    ) as check2:
        pass

    pytest_xfail.any_failed(check1, check2)


def test_decorator_usage() -> None:
    @pytest_xfail("lib/#1234", raises=AssertionError)
    def test_func1() -> None:
        raises_assertionerror()

    @pytest_xfail("lib/#4567", raises=ValueError)
    def test_func2() -> None:
        raises_runtimeerror()

    with pytest.raises(XFailed):
        test_func1()

    with pytest.raises(RuntimeError):
        test_func2()
