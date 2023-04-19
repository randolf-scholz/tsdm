"""Context managers for use in decorators."""

__all__ = [
    "ray_cluster",
]

import importlib
import logging
import os
from contextlib import ContextDecorator
from types import ModuleType, TracebackType
from typing import ClassVar, Optional

from typing_extensions import Self

if __name__ == "__main__":
    # main program
    pass


class ray_cluster(ContextDecorator):
    """Context manager for starting and stopping a ray cluster."""

    LOGGER: ClassVar[logging.Logger] = logging.getLogger(__qualname__)
    """Logger for this class."""
    ray: ModuleType | None = None
    """Ray module."""
    num_cpus: int
    """Number of CPUs to use for the ray cluster."""

    def __init__(self, num_cpus: Optional[int] = None) -> None:
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
    ) -> bool:
        self.LOGGER.warning("Tearing down ray cluster.")

        if self.ray is not None:
            self.LOGGER.warning("Tearing down ray cluster.")
            self.ray.shutdown()
        return False
