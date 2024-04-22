r"""Submodule containing general purpose decorators.

#TODO add module description.
"""

__all__ = [
    # Protocols
    "Decorator",
    "ClassDecorator",
    # Constants
    "DECORATORS",
    # Functions
    "debug",
    "decorator",
    "return_namedtuple",
    "timefun",
    "trace",
    "vectorize",
    "wrap_func",
    "wrap_method",
    # "sphinx_value",
    # context managers
]


from tsdm.utils.decorators._decorators import (
    ClassDecorator,
    Decorator,
    debug,
    decorator,
    implements,
    return_namedtuple,
    timefun,
    trace,
    vectorize,
    wrap_func,
    wrap_method,
)

DECORATORS: dict[str, Decorator] = {
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
    "implements": implements,
}
