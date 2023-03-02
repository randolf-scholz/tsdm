r"""Logging Utility Functions."""

__all__ = [
    # Constants
    "Logger",
    "LOGGERS",
    # Functions
    "compute_metrics",
    "log_optimizer_state",
    "log_kernel_state",
    "log_model_state",
    "log_metrics",
    "log_scalars",
    # Classes
]

from collections.abc import Callable
from typing import TypeAlias

from tsdm.logutils._callbacks import (
    compute_metrics,
    log_kernel_state,
    log_metrics,
    log_model_state,
    log_optimizer_state,
    log_scalars,
)
from tsdm.logutils.base import StandardLogger

Logger: TypeAlias = Callable[..., None]

LOGGERS: dict[str, Logger] = {
    "log_optimizer_state": log_optimizer_state,
    "log_kernel_information": log_kernel_state,
    "log_model_state": log_model_state,
    "log_metrics": log_metrics,
}

del TypeAlias, Callable
