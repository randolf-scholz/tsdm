r"""Module Summary Line.

Module description
"""

from __future__ import annotations

__all__ = [
    "log_optimizer_state",
    "log_kernel_information",
    "log_model_state",
    "log_metrics",
]

import logging
from typing import Callable, Final

from tsdm.util.logging._logging import (
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
)

LOGGER = logging.getLogger(__name__)


Logger = Callable[..., int]

LOGGERS: Final[dict[str, Logger]] = {
    "log_optimizer_state": log_optimizer_state,
    "log_kernel_information": log_kernel_information,
    "log_model_state": log_model_state,
    "log_metrics": log_metrics,
}
