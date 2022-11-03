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
    "wrap_func",
    "wrap_method",
]

from collections.abc import Callable
from typing import Final, TypeAlias

from tsdm.utils.decorators._decorators import (
    IterItems,
    IterKeys,
    autojit,
    decorator,
    timefun,
    trace,
    vectorize,
    wrap_func,
    wrap_method,
)

Decorator: TypeAlias = Callable[..., Callable]
r"""Type hint for dataset."""

DECORATORS: Final[dict[str, Decorator]] = {
    "IterItems": IterItems,
    "IterKeys": IterKeys,
    "autojit": autojit,
    "decorator": decorator,
    # "sphinx_value": sphinx_value,
    "timefun": timefun,
    "trace": trace,
    "vectorize": vectorize,
    "wrap_func": wrap_func,
    "wrap_method": wrap_method,
}
r"""Dictionary of all available decorators."""

del Final, TypeAlias, Callable
