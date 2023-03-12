r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

__all__ = [
    # Classes
    # Constants
    "Decorator",
    "DECORATORS",
    # Functions
    "IterItems",
    "IterKeys",
    "autojit",
    "decorator",
    "return_namedtuple",
    "timefun",
    "trace",
    "vectorize",
    "wrap_func",
    "wrap_method",
    # "sphinx_value",
]

from collections.abc import Callable
from typing import Final, TypeAlias

from tsdm.utils.decorators._decorators import (
    IterItems,
    IterKeys,
    autojit,
    decorator,
    return_namedtuple,
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
    "named_return": return_namedtuple,
    "timefun": timefun,
    "trace": trace,
    "vectorize": vectorize,
    "wrap_func": wrap_func,
    "wrap_method": wrap_method,
    # "sphinx_value": sphinx_value,
}
r"""Dictionary of all available decorators."""

del Final, TypeAlias, Callable
