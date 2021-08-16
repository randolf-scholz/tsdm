r"""Module Docstring."""

import logging
from typing import Final

from tsdm.random.utils import sample_timestamps, sample_timedeltas

logger = logging.getLogger(__name__)
__all__: Final[list[str]] = ["sample_timestamps", "sample_timedeltas"]
