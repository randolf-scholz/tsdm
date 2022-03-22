r"""Logging Utility Functions."""

__all__ = [
    # Constants
    "Logger",
    "LOGGERS",
    # Functions
    "compute_metrics",
    "log_optimizer_state",
    "log_kernel_information",
    "log_model_state",
    "log_metrics",
]

import logging
from collections.abc import Callable
from typing import Final

from tsdm.logutils._logutils import (
    compute_metrics,
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
)

__logger__ = logging.getLogger(__name__)


Logger = Callable[..., None]

LOGGERS: Final[dict[str, Logger]] = {
    "log_optimizer_state": log_optimizer_state,
    "log_kernel_information": log_kernel_information,
    "log_model_state": log_model_state,
    "log_metrics": log_metrics,
}
