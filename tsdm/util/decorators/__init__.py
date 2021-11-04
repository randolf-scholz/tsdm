r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

from __future__ import annotations

__all__ = [
    # Classes
    # Constants
    "Decorator",
    "DECORATORS",
    # Functions
    "autojit",
    "decorator",
    "sphinx_value",
    "timefun",
    "trace",
]

import logging
from typing import Callable, Final

from tsdm.util.decorators._decorators import (
    autojit,
    decorator,
    sphinx_value,
    timefun,
    trace,
)

__logger__ = logging.getLogger(__name__)

Decorator = Callable[..., Callable]
r"""Type hint for datasets."""

DECORATORS: Final[dict[str, Decorator]] = {
    "autojit": autojit,
    "decorator": decorator,
    "sphinx_value": sphinx_value,
    "timefun": timefun,
    "trace": trace,
}
r"""Dictionary of all available decorators."""
