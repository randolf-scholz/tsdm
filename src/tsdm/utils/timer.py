r"""A timer context manager."""

__all__ = ["timer"]

import gc
import logging
import signal
import sys
from contextlib import ContextDecorator
from time import perf_counter_ns
from types import FrameType, TracebackType
from typing import ClassVar, Never, Self, cast, overload


class timer[ExitT: bool | None = None](ContextDecorator):
    r"""Context manager for timing a block of code."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")

    start_time: int
    r"""Start time of the timer."""
    end_time: int
    r"""End time of the timer."""
    disable_gc: bool = False
    r"""Whether to disable garbage collection."""
    timeout: float | None = None
    r"""Timeout in seconds."""

    @overload
    def __init__(
        self: timer[None], timeout: None = ..., *, disable_gc: bool = ...
    ) -> None: ...
    @overload
    def __init__(
        self: timer[bool], timeout: float, *, msg: str = ..., disable_gc: bool = ...
    ) -> None: ...
    def __init__(
        self,
        timeout: float | None = None,
        *,
        msg: str = "Execution timed out.",
        disable_gc: bool = False,
    ) -> None:
        super().__init__()
        self.disable_gc = disable_gc
        self.timeout = timeout
        self.exception = TimeoutError(msg)

    def _timeout_handler(self, signum: int, frame: FrameType | None, /) -> Never:  # noqa: ARG002
        self.exception.add_note(f"Timed out after {self.timeout} seconds.")
        raise self.exception

    def __enter__(self) -> Self:
        r"""Disable garbage collection and start the timer."""
        # flush pending writes
        sys.stdout.flush()
        sys.stderr.flush()

        # collect garbage
        gc.collect()

        # disable garbage collection
        if self.disable_gc:
            gc.disable()

        if self.timeout is not None:
            assert self.timeout > 0  # timeout must be positive
            # Save previous state (supports nesting reasonably well)
            self._old_handler = signal.getsignal(signal.SIGALRM)
            self._old_itimer = signal.getitimer(signal.ITIMER_REAL)
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, self.timeout)

        # start timer
        self.start_time = perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> ExitT:
        r"""Stop the timer and re-enable garbage collection."""
        self.end_time = perf_counter_ns()

        if self.timeout is not None:
            # Cancel the scheduled alarm
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            # Restore previous signal handler and itimer
            signal.signal(signal.SIGALRM, self._old_handler)
            signal.setitimer(
                signal.ITIMER_REAL,
                self._old_itimer[0],
                self._old_itimer[1],
            )

        if self.disable_gc:
            gc.enable()

        return cast(
            "ExitT", None if self.timeout is None else (exc_val is self.exception)
        )

    @property
    def remaining_time(self) -> float | None:
        r"""Remaining time in seconds."""
        if self.timeout is None:
            return None
        return self.timeout - self.elapsed_seconds

    @property
    def elapsed_time(self) -> int:
        r"""Elapsed time in nanoseconds."""
        if start_time := getattr(self, "start_time", None) is None:
            raise RuntimeError("Timer has not been started!")
        if end_time := getattr(self, "end_time", None) is None:
            return perf_counter_ns() - start_time
        return end_time - start_time

    @property
    def elapsed_seconds(self) -> float:
        r"""Elapsed time in seconds."""
        return self.elapsed_time / 1_000_000_000

    @property
    def value(self) -> str:
        r"""Formatted elapsed time."""
        return _format_ns(self.elapsed_time)


def _format_ns(ns: int, /) -> str:
    r"""Format nanoseconds into a human-readable string."""
    hours, remainder = divmod(ns, 3_600_000_000_000)
    minutes, remainder = divmod(remainder, 60_000_000_000)
    seconds, remainder = divmod(remainder, 1_000_000_000)
    milliseconds, remainder = divmod(remainder, 1_000_000)
    microseconds = remainder // 1_000

    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    if seconds:  # print 2 decimal places
        return f"{seconds}.{remainder // 10**7:02d}s"
    if milliseconds:  # print 2 decimal places
        return f"{milliseconds}.{remainder // 10**4:02d}ms"
    if microseconds:  # print 2 decimal places
        return f"{microseconds}.{remainder // 10}µs"
    return f"{remainder}ns"
