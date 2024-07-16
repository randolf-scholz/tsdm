r"""Context managers for use in decorators."""

__all__ = [
    # Protocol
    # use contextlib.AbstractContextManager instead
    # Classes
    "ray_cluster",
    "system_path",
    "timeout",
    "timer",
]

import gc
import importlib
import logging
import os
import signal
import sys
from contextlib import AbstractContextManager, ContextDecorator
from dataclasses import KW_ONLY, dataclass
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter_ns
from types import FrameType, ModuleType, TracebackType

from typing_extensions import ClassVar, Literal, Never, Optional, Self


class ray_cluster(ContextDecorator):
    r"""Context manager for starting and stopping a ray cluster."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")
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
    ) -> Literal[False]:
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
    ) -> Literal[False]:
        sys.path = self.previous_path
        return False


class timer(ContextDecorator):
    r"""Context manager for timing a block of code."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__name__}/{__qualname__}")

    start_time: int
    r"""Start time of the timer."""
    end_time: int
    r"""End time of the timer."""
    elapsed_time: int
    r"""Elapsed time of the timer in nano-seconds."""
    elapsed_seconds: float
    r"""Elapsed time of the timer in seconds."""

    def __enter__(self) -> Self:
        # flush pending writes
        sys.stdout.flush()
        sys.stderr.flush()
        # disable garbage collection
        gc.collect()
        gc.disable()
        # start timer
        self.start_time = perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> Literal[False]:
        # stop timer
        self.end_time = perf_counter_ns()
        self.elapsed_time = self.end_time - self.start_time
        self.elapsed_seconds = self.elapsed_time / 1_000_000_000
        # re-enable garbage collection
        gc.enable()
        gc.collect()
        return False

    @property
    def value(self) -> str:
        r"""Formatted elapsed time."""
        hours, remainder = divmod(self.elapsed_time, 3_600_000_000_000)
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


@dataclass
class timeout(ContextDecorator, AbstractContextManager):
    r"""Context manager for timing out a block of code."""

    num_seconds: int

    _: KW_ONLY

    suppress: bool = False

    def __post_init__(self):
        self._exception = TimeoutError("Execution timed out.")

    def _timeout_handler(self, signum: int, frame: None | FrameType) -> Never:  # noqa: ARG002
        raise self._exception

    def __enter__(self) -> Self:
        # Set the signal handler for SIGALRM (alarm signal)
        signal.signal(signal.SIGALRM, self._timeout_handler)
        # Schedule the alarm to go off in num_seconds seconds
        signal.alarm(self.num_seconds)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> bool:
        # Cancel the scheduled alarm
        signal.alarm(0)
        # Reset the signal handler to its default behavior
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        if exc_type is self._exception:
            return self.suppress
        return False
