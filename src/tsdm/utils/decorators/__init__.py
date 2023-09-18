r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

__all__ = [
    # Protocols
    "Decorator",
    "ClassDecorator",
    # Constants
    "DECORATORS",
    "CONTEXT_MANAGERS",
    "CLASS_DECORATORS",
    # Functions
    "IterItems",
    "IterKeys",
    "IterValues",
    "debug",
    "autojit",
    "decorator",
    "return_namedtuple",
    "timefun",
    "trace",
    "vectorize",
    "wrap_func",
    "wrap_method",
    # "sphinx_value",
    # context managers
    "ray_cluster",
    "timer",
]

from contextlib import AbstractContextManager

from tsdm.utils.decorators._contextmanagers import ray_cluster, timer
from tsdm.utils.decorators._decorators import (
    ClassDecorator,
    Decorator,
    IterItems,
    IterKeys,
    IterValues,
    autojit,
    debug,
    decorator,
    return_namedtuple,
    timefun,
    trace,
    vectorize,
    wrap_func,
    wrap_method,
)

CONTEXT_MANAGERS: dict[str, type[AbstractContextManager]] = {
    "ray_cluster": ray_cluster,
    "timer": timer,
}
r"""Dictionary of all available context managers."""

DECORATORS: dict[str, Decorator] = {
    "IterItems": IterItems,
    "IterKeys": IterKeys,
    "IterValues": IterValues,
    "decorator": decorator,
    "debug": debug,
    "named_return": return_namedtuple,
    "timefun": timefun,
    "trace": trace,
    "vectorize": vectorize,
    "wrap_func": wrap_func,
    "wrap_method": wrap_method,
    # "sphinx_value": sphinx_value,
}
r"""Dictionary of all available decorators."""


CLASS_DECORATORS: dict[str, ClassDecorator] = {
    "autojit": autojit,
}
r"""Dictionary of all available class decorators."""

del AbstractContextManager
