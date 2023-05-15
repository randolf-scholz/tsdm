r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

__all__ = [
    # Protocols
    "Decorator",
    "ClassDecorator",
    "ContextManager",
    # Constants
    "DECORATORS",
    "CONTEXT_MANAGERS",
    "CLASS_DECORATORS",
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
    # context managers
    "ray_cluster",
    "timer",
]

from tsdm.utils.decorators._contextmanagers import ContextManager, ray_cluster, timer
from tsdm.utils.decorators._decorators import (
    ClassDecorator,
    Decorator,
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

CONTEXT_MANAGERS: dict[str, type[ContextManager]] = {
    "ray_cluster": ray_cluster,
    "timer": timer,
}
r"""Dictionary of all available context managers."""

DECORATORS: dict[str, Decorator] = {
    "IterItems": IterItems,
    "IterKeys": IterKeys,
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


CLASS_DECORATORS: dict[str, ClassDecorator] = {
    "autojit": autojit,
}
r"""Dictionary of all available class decorators."""

del ContextManager
