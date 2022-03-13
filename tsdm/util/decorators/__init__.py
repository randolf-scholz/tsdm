r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

__all__ = [
    # Classes
    # Constants
    "Decorator",
    "DECORATORS",
    # Functions
    "autojit",
    "decorator",
    # "sphinx_value",
    "timefun",
    "trace",
    "vectorize",
    "IterItems",
    "IterKeys",
    "pre_hook",
    "post_hook",
    "wrap_hook",
    "wrapmethod",
]

import logging
from collections.abc import Callable
from typing import Final

from tsdm.util.decorators._decorators import (  # sphinx_value,
    IterItems,
    IterKeys,
    autojit,
    decorator,
    post_hook,
    pre_hook,
    timefun,
    trace,
    vectorize,
    wrap_hook,
    wrapmethod,
)

__logger__ = logging.getLogger(__name__)

Decorator = Callable[..., Callable]
r"""Type hint for dataset."""

DECORATORS: Final[dict[str, Decorator]] = {
    "autojit": autojit,
    "decorator": decorator,
    # "sphinx_value": sphinx_value,
    "timefun": timefun,
    "trace": trace,
}
r"""Dictionary of all available decorators."""
