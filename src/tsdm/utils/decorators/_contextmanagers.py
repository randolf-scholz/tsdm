"""Context managers for use in decorators."""

__all__ = [
    # Protocol
    "ContextManager",
    # Classes
    "ray_cluster",
    "timer",
]

import gc
import importlib
import logging
import os
import sys
from contextlib import ContextDecorator
from time import perf_counter_ns
from types import ModuleType, TracebackType
from typing import ClassVar, ContextManager, Literal, Optional

from typing_extensions import Self


class ray_cluster(ContextDecorator):
    """Context manager for starting and stopping a ray cluster."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__module__}/{__qualname__}")  # type: ignore[name-defined]
    """Logger for this class."""
    ray: ModuleType | None = None
    """Ray module."""
    num_cpus: int
    """Number of CPUs to use for the ray cluster."""

    def __init__(self, *, num_cpus: Optional[int] = None) -> None:
        super().__init__()
        self.num_cpus = (
            max(1, ((os.cpu_count() or 0) * 4) // 5) if num_cpus is None else num_cpus
        )

    def __enter__(self) -> Self:
        if importlib.util.find_spec("ray") is not None:
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
    ) -> Literal[False]:
        self.LOGGER.warning("Tearing down ray cluster.")

        if self.ray is not None:
            self.LOGGER.warning("Tearing down ray cluster.")
            self.ray.shutdown()
        return False


class timer(ContextDecorator):
    """Context manager for timing a block of code."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(f"{__module__}/{__qualname__}")  # type: ignore[name-defined]

    start_time: int
    """Start time of the timer."""
    end_time: int
    """End time of the timer."""
    elapsed: float
    """Elapsed time of the timer in seconds."""

    def __enter__(self) -> Self:
        self.LOGGER.info("Flushing pending writes.")
        sys.stdout.flush()
        sys.stderr.flush()
        self.LOGGER.info("Disabling garbage collection.")
        gc.collect()
        gc.disable()
        self.LOGGER.info("Starting timer.")
        self.start_time = perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.end_time = perf_counter_ns()
        self.elapsed = (self.end_time - self.start_time) / 10**9
        self.LOGGER.info("Stopped timer.")
        gc.enable()
        self.LOGGER.info("Re-Enabled garbage collection.")
        return False
