r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    "trace",
    # Constants
    "Decorator",
    "DECORATORS",
    # Functions
    "decorator",
    "timefun",
    "sphinx_value",
]

import logging
from typing import Callable, Final

from tsdm.util.decorators.decorators import decorator, sphinx_value, timefun, trace

LOGGER = logging.getLogger(__name__)

Decorator = Callable[..., Callable]
r"""Type hint for datasets."""

DECORATORS: Final[dict[str, Decorator]] = {
    "decorator": decorator,
    "timefun": timefun,
    "sphinx_value": sphinx_value,
}
r"""Dictionary of all available decorators."""
