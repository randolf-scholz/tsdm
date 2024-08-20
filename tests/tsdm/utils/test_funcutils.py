r"""Tests for `tsdm.utils.funcutils` module."""
# ruff: noqa: ARG001

from tsdm.utils.funcutils import (
    is_keyword_arg,
    is_keyword_only_arg,
    is_mandatory_arg,
    is_positional_arg,
    is_positional_only_arg,
    is_variadic_arg,
)


def foo(a, /): ...  # type: ignore[no-untyped-def]
def bar(a): ...  # type: ignore[no-untyped-def]
def baz(*a): ...  # type: ignore[no-untyped-def]
def qux(*, a): ...  # type: ignore[no-untyped-def]
def quux(**a): ...  # type: ignore[no-untyped-def]
def foo2(a=None, /): ...  # type: ignore[no-untyped-def]
def bar2(a=None): ...  # type: ignore[no-untyped-def]
def qux2(*, a=None): ...  # type: ignore[no-untyped-def]


def test_is_mandatory_arg() -> None:
    assert is_mandatory_arg(foo, "a") is True
    assert is_mandatory_arg(bar, "a") is True
    assert is_mandatory_arg(baz, "a") is False
    assert is_mandatory_arg(qux, "a") is True
    assert is_mandatory_arg(quux, "a") is False

    assert is_mandatory_arg(foo2, "a") is False
    assert is_mandatory_arg(bar2, "a") is False
    assert is_mandatory_arg(qux2, "a") is False


def test_is_positional_arg() -> None:
    assert is_positional_arg(foo, "a") is True
    assert is_positional_arg(bar, "a") is True
    assert is_positional_arg(baz, "a") is True
    assert is_positional_arg(qux, "a") is False
    assert is_positional_arg(quux, "a") is False

    assert is_positional_arg(foo2, "a") is True
    assert is_positional_arg(bar2, "a") is True
    assert is_positional_arg(qux2, "a") is False


def test_is_positional_only_arg() -> None:
    assert is_positional_only_arg(foo, "a") is True
    assert is_positional_only_arg(bar, "a") is False
    assert is_positional_only_arg(baz, "a") is True
    assert is_positional_only_arg(qux, "a") is False
    assert is_positional_only_arg(quux, "a") is False

    assert is_positional_only_arg(foo2, "a") is True
    assert is_positional_only_arg(bar2, "a") is False
    assert is_positional_only_arg(qux2, "a") is False


def test_is_keyword_arg() -> None:
    assert is_keyword_arg(foo, "a") is False
    assert is_keyword_arg(bar, "a") is True
    assert is_keyword_arg(baz, "a") is False
    assert is_keyword_arg(qux, "a") is True
    assert is_keyword_arg(quux, "a") is True

    assert is_keyword_arg(foo2, "a") is False
    assert is_keyword_arg(bar2, "a") is True
    assert is_keyword_arg(qux2, "a") is True


def test_is_keyword_only_arg() -> None:
    assert is_keyword_only_arg(foo, "a") is False
    assert is_keyword_only_arg(bar, "a") is False
    assert is_keyword_only_arg(baz, "a") is False
    assert is_keyword_only_arg(qux, "a") is True
    assert is_keyword_only_arg(quux, "a") is True

    assert is_keyword_only_arg(foo2, "a") is False
    assert is_keyword_only_arg(bar2, "a") is False
    assert is_keyword_only_arg(qux2, "a") is True


def test_is_variadic_arg() -> None:
    assert is_variadic_arg(foo, "a") is False
    assert is_variadic_arg(bar, "a") is False
    assert is_variadic_arg(baz, "a") is True
    assert is_variadic_arg(qux, "a") is False
    assert is_variadic_arg(quux, "a") is True

    assert is_variadic_arg(foo2, "a") is False
    assert is_variadic_arg(bar2, "a") is False
    assert is_variadic_arg(qux2, "a") is False
