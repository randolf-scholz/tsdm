r"""Tests for tsdm.

NOTE: We use `/tests/tsdm/...` layout to ensure that the tests are not imported.
"""

__all__ = ["pytest_xfail"]

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from functools import wraps
from types import TracebackType

import pytest


class pytest_xfail(AbstractContextManager):
    r"""Context manager for marking code as expected to fail.

    Usage:  (as context manager)
        ```python
        with pytest_xfail("reason for xfail", strict=True):
            # code that is expected to fail
        ```

    Usage:  (as decorator)
        ```python
        @pytest.mark.parametrize("x", [1, 2, 3, 4, 5])
        @pytest_xfail("reason for xfail", condition=lambda x: x > 3)
        def test_function(x: int) -> None:
            assert x <= 3
        ```
    """

    def __bool__(self) -> bool:
        # True if error, or no error and strict mode is enabled.
        return self.triggered

    def __init__(
        self,
        reason: str = "",
        *,
        strict: bool = True,
        raises: Sequence[type[BaseException]] | type[BaseException] | None = None,
        defer_xfail: bool = False,
        condition: Callable[..., bool] | bool | None = None,
    ) -> None:
        self.strict: bool = strict
        self.reason: str = reason
        self.failed: bool = NotImplemented
        self.raises: tuple[type[BaseException], ...] | None = (
            None
            if raises is None
            else ((raises,) if isinstance(raises, type) else tuple(raises))
        )
        self.defer_xfail: bool = defer_xfail
        self.exc_type: type[BaseException] | None = None
        self.exc_value: BaseException | None = None
        self.traceback: TracebackType | None = None
        self.condition: Callable[..., bool] | bool | None = condition
        self.triggered: bool = False

    def __call__[**P](self, func: Callable[P, None], /) -> Callable[P, None]:
        r"""Decorator version of the context manager."""

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
            if (
                self.condition is None
                or self.condition is True
                or (callable(self.condition) and self.condition(*args, **kwargs))
            ):
                with self:
                    func(*args, **kwargs)
            else:
                func(*args, **kwargs)

        return wrapper

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool:
        if self.condition is False:
            return False  # do nothing

        self.exc_type = exc_type
        self.exc_value = exc_value
        self.traceback = traceback

        # no exception raised
        if exc_type is None:
            if self.strict:
                self.triggered = True
                raise AssertionError("Expected test to fail, but it passed.")
            return True

        # caught expected exception
        if self.raises is None or any(issubclass(exc_type, r) for r in self.raises):
            self.triggered = True
            if not self.defer_xfail:
                pytest.xfail(
                    f"{self.reason}\n Due to: {exc_type.__name__}: {exc_value}"
                )
            return True

        # caught unexpected exception, re-raise
        assert exc_value is not None
        raise exc_value

    @staticmethod
    def any_failed(*cms: pytest_xfail) -> None:
        r"""Check if any of the context managers in `it` are active."""
        if not any(cms):
            return

        # at least one context manager xfailed
        reason = "\n".join(f"{i}: {cm.reason}" for i, cm in enumerate(cms))
        pytest.xfail(reason)
