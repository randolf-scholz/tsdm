r"""Tests for tsdm.

NOTE: We use `/tests/tsdm/...` layout to ensure that the tests are not imported.
"""

__all__ = ["pytest_xfail"]

from contextlib import AbstractContextManager
from types import TracebackType

import pytest


class pytest_xfail(AbstractContextManager):
    r"""Context manager for marking code as expected to fail."""

    def __bool__(self) -> bool:
        # True if error, or no error and strict mode is enabled.
        return self.exc_type is not None or self.strict

    def __init__(
        self, reason: str, *, strict: bool = False, raise_on_exit: bool = True
    ) -> None:
        self.strict: bool = strict
        self.reason: str = reason
        self.failed: bool = NotImplemented
        self.raise_on_exit: bool = raise_on_exit
        self.exc_type: type[BaseException] | None = None
        self.exc_value: BaseException | None = None
        self.traceback: TracebackType | None = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool:
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.traceback = traceback

        if exc_type is None:
            if self.strict:
                raise AssertionError("Expected test to fail, but it passed.")
            return True
        if self.raise_on_exit:
            pytest.xfail(f"{self.reason}\n Due to: {exc_type.__name__}: {exc_value}")
        return True

    @staticmethod
    def raise_if_any(*cms: "pytest_xfail") -> None:
        r"""Check if any of the context managers in `it` are active."""
        if not any(cms):
            return

        # at least one context manager xfailed
        reason = "\n".join(f"{i}: {cm.reason}" for i, cm in enumerate(cms))
        pytest.xfail(reason)
