r"""Submodule containing general purpose decorators."""

__all__ = [
    # Constants
    "DECORATORS",
    "CLASS_DECORATORS",
    # Protocols
    "Decorator",
    "ClassDecorator",
    # Functions
    "autojit",
    "debug",
    "decorator",
    "implements",
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
    autojit,
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
    "decorator"    : decorator,
    "debug"        : debug,
    "named_return" : return_namedtuple,
    "timefun"      : timefun,
    "trace"        : trace,
    "vectorize"    : vectorize,
    "wrap_func"    : wrap_func,
    "wrap_method"  : wrap_method,
    # "sphinx_value": sphinx_value,
}  # fmt: skip
r"""Dictionary of all available decorators."""

CLASS_DECORATORS: dict[str, ClassDecorator] = {
    "implements": implements,
    "autojit"   : autojit,
}  # fmt: skip
r"""Dictionary of all available class decorators."""
