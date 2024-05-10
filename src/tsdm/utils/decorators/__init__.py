r"""Submodule containing general purpose decorators."""

__all__ = [
    # Constants
    "CLASS_DECORATORS",
    "CLASS_DECORATOR_FACTORIES",
    "FUNCTION_DECORATORS",
    # Protocols
    "ClassDecorator",
    "ClassDecoratorFactory",
    "FunctionDecorator",
    "FunctionDecoratorFactory",
    # Functions
    "autojit",
    "debug",
    "decorator",
    "implements",
    "pprint_dataclass",
    "pprint_mapping",
    "pprint_namedtuple",
    "pprint_repr",
    "pprint_sequence",
    "return_namedtuple",
    "timefun",
    "trace",
    "vectorize",
    "wrap_func",
    "wrap_method",
]


from tsdm.utils.decorators._decorators import (
    decorator,
)
from tsdm.utils.decorators.class_decorators import (
    ClassDecorator,
    ClassDecoratorFactory,
    autojit,
    implements,
    pprint_dataclass,
    pprint_mapping,
    pprint_namedtuple,
    pprint_repr,
    pprint_sequence,
)
from tsdm.utils.decorators.func_decorators import (
    FunctionDecorator,
    FunctionDecoratorFactory,
    debug,
    return_namedtuple,
    timefun,
    trace,
    vectorize,
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
    "vectorize"        : vectorize,
}  # fmt: skip
r"""Dictionary of all available function decorators."""

CLASS_DECORATORS: dict[str, ClassDecorator] = {
    "autojit"          : autojit,
    "pprint_dataclass" : pprint_dataclass,
    "pprint_mapping"   : pprint_mapping,
    "pprint_repr"      : pprint_repr,
    "pprint_sequence"  : pprint_sequence,
    "pprint_namedtuple": pprint_sequence,
}  # fmt: skip
r"""Dictionary of all available class decorators."""

CLASS_DECORATOR_FACTORIES: dict[str, ClassDecoratorFactory] = {
    "implements": implements,
}  # fmt: skip
r"""Dictionary of all available class decorator factories."""
