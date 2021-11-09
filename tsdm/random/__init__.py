r"""TODO: Module Docstring.

TODO: Module Docstring
"""

from __future__ import annotations

__all__ = [
    # Functions
    "sample_timestamps",
    "sample_timedeltas",
]


import logging

from tsdm.random.utils import sample_timedeltas, sample_timestamps

__logger__ = logging.getLogger(__name__)
