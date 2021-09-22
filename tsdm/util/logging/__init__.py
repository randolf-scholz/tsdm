r"""Module Summary Line.

Module description
"""  # pylint: disable=line-too-long # noqa

from __future__ import annotations

import logging
from typing import Callable, Final

from tsdm.util.logging._logging import log_kernel_information, log_optimizer_state

LOGGER = logging.getLogger(__name__)

__all__: Final[list[str]] = ["log_optimizer_state", "log_kernel_information"]

Logger = Callable[..., int]

LOGGERS: Final[dict[str, Logger]] = {
    "log_optimizer_state": log_optimizer_state,
    "log_kernel_information": log_kernel_information,
}
