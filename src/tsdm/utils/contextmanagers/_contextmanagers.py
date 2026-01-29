r"""Context managers for use in decorators."""

__all__ = [
    # Protocol
    "ContextManager",
    # Classes
    "ray_cluster",
    "system_path",
    "timer",
]

import gc
import importlib
import logging
import os
import signal
import sys
from contextlib import AbstractContextManager as ContextManager, ContextDecorator
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter_ns
from types import FrameType, ModuleType, TracebackType
from typing import ClassVar, Literal as L, Never, Optional, Self, cast, overload


class ray_cluster(ContextDecorator):
    r"""Context manager for starting and stopping a ray cluster."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}.{__qualname__}")
    r"""Logger for this class."""
    ray: ModuleType | None = None
    r"""Ray module."""
    num_cpus: int
    r"""Number of CPUs to use for the ray cluster."""

    def __init__(self, *, num_cpus: Optional[int] = None) -> None:
        super().__init__()
        self.num_cpus = (
            max(1, ((os.cpu_count() or 0) * 4) // 5) if num_cpus is None else num_cpus
        )

    def __enter__(self) -> Self:
        if find_spec("ray") is not None:
            self.ray = importlib.import_module("ray")
            # Only use 80% of the available CPUs.
            self.LOGGER.warning("Starting ray cluster with num_cpus=%s.", self.num_cpus)
            self.ray.init(num_cpus=self.num_cpus)
        else:
            self.LOGGER.warning("Ray not found, skipping ray cluster.")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> L[False]:
        self.LOGGER.warning("Tearing down ray cluster.")

        if self.ray is not None:
            self.LOGGER.warning("Tearing down ray cluster.")
            self.ray.shutdown()
        return False


class system_path(ContextDecorator):
    r"""Prepends a path to environment variable `$PATH`.

    References:
        - https://stackoverflow.com/a/41904558
    """

    path: Path
    previous_path: list[str]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist.")

    def __enter__(self) -> Self:
        self.previous_path = sys.path.copy()
        sys.path.insert(0, str(self.path))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> L[False]:
        sys.path = self.previous_path
        return False


class timer[ExitT: bool](ContextDecorator):
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
    def __init__(self: timer[L[False]], *, disable_gc: bool = ...) -> None: ...
    @overload
    def __init__(
        self: timer[bool],
        timeout: float | None,
        *,
        msg: str = ...,
        disable_gc: bool = ...,
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

    def _timeout_handler(self, signum: int, frame: FrameType | None) -> Never:  # noqa: ARG002
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

        return cast("ExitT", exc_val is self.exception)

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
