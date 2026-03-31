r"""Submodule containing general purpose decorators."""

__all__ = [
    # Constants
    "CLASS_DECORATORS",
    "CLASS_DECORATOR_FACTORIES",
    "FUNCTION_DECORATORS",
    # Protocols & ABCs
    "ClassDecorator",
    "ClassDecoratorFactory",
    "Decorator",
    "DecoratorFactory",
    "FunctionDecorator",
    "FunctionDecoratorFactory",
    "ParametrizedClassDecorator",
    "ParametrizedDecorator",
    "ParametrizedFunctionDecorator",
    # Functions
    "debug",
    "decorator",
    "implements",
    "pprint_dataclass",
    "pprint_mapping",
    "pprint_namedtuple",
    "pprint_repr",
    "pprint_sequence",
    "pprint_set",
    "recurse_on_container",
    "return_namedtuple",
    "timefun",
    "trace",
    "wrap_func",
    "wrap_method",
]


from .base import (
    ClassDecorator,
    ClassDecoratorFactory,
    Decorator,
    DecoratorFactory,
    FunctionDecorator,
    FunctionDecoratorFactory,
    ParametrizedClassDecorator,
    ParametrizedDecorator,
    ParametrizedFunctionDecorator,
    decorator,
    recurse_on_container,
)
from .class_decorators import (
    implements,
    pprint_dataclass,
    pprint_mapping,
    pprint_namedtuple,
    pprint_repr,
    pprint_sequence,
    pprint_set,
)
from .func_decorators import (
    debug,
    return_namedtuple,
    timefun,
    trace,
    wrap_func,
    wrap_method,
)

FUNCTION_DECORATORS: dict[str, FunctionDecorator] = {
    "debug"            : debug,
    "return_namedtuple": return_namedtuple,
    "timefun"          : timefun,
    "trace"            : trace,
    "wrap_func"        : wrap_func,
    "wrap_method"      : wrap_method,
}  # fmt: skip
r"""Dictionary of all available function decorators."""

CLASS_DECORATORS: dict[str, ClassDecorator] = {
    "pprint_dataclass" : pprint_dataclass,
    "pprint_mapping"   : pprint_mapping,
    "pprint_repr"      : pprint_repr,
    "pprint_sequence"  : pprint_sequence,
    "pprint_namedtuple": pprint_namedtuple,
    "pprint_set"       : pprint_set,
}  # fmt: skip
r"""Dictionary of all available class decorators."""

CLASS_DECORATOR_FACTORIES: dict[str, ClassDecoratorFactory] = {
    "implements": implements,
}  # fmt: skip
r"""Dictionary of all available class decorator factories."""
