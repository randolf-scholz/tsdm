r"""TODO: Module Docstring.

TODO: Module summary.
"""

from __future__ import annotations

__all__ = [
    # Constants
    "Logger",
    "__logger__S",
    # Functions
    "compute_metrics",
    "log_optimizer_state",
    "log_kernel_information",
    "log_model_state",
    "log_metrics",
]

import logging
from typing import Callable, Final

from tsdm.logutils._logutils import (
    compute_metrics,
    log_kernel_information,
    log_metrics,
    log_model_state,
    log_optimizer_state,
)

__logger__ = logging.getLogger(__name__)


Logger = Callable[..., int]

__logger__S: Final[dict[str, Logger]] = {
    "log_optimizer_state": log_optimizer_state,
    "log_kernel_information": log_kernel_information,
    "log_model_state": log_model_state,
    "log_metrics": log_metrics,
}
