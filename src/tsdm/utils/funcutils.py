"""Utility functions relating to function arguments and return values."""

__all__ = [
    "accepts_varkwargs",
    "dataclass_args_kwargs",
    "get_function_args",
    "get_parameter_kind",
    "is_keyword_arg",
    "is_keyword_only_arg",
    "is_mandatory_arg",
    "is_positional_arg",
    "is_positional_only_arg",
    "is_variadic_arg",
    "get_mandatory_argcount",
    "get_mandatory_kwargs",
    "get_return_typehint",
]

import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from inspect import Parameter

from typing_extensions import Any, Optional, ParamSpec, overload

from tsdm.types.protocols import Dataclass, is_dataclass
from tsdm.types.variables import R

KEYWORD_ONLY = Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = Parameter.VAR_KEYWORD
VAR_POSITIONAL = Parameter.VAR_POSITIONAL
Kind = inspect._ParameterKind  # pylint: disable=protected-access

PARAMETER_KINDS = {
    "positional": {POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL},
    "positional_only": {POSITIONAL_ONLY, VAR_POSITIONAL},
    "keyword": {POSITIONAL_OR_KEYWORD, KEYWORD_ONLY, VAR_KEYWORD},
    "keyword_only": {KEYWORD_ONLY, VAR_KEYWORD},
    "variadic": {VAR_POSITIONAL, VAR_KEYWORD},
}

P = ParamSpec("P")


def rpartial(
    func: Callable[P, R], /, *fixed_args: Any, **fixed_kwargs: Any
) -> Callable[..., R]:
    r"""Apply positional arguments from the right.

    References:
        - https://docs.python.org/3/library/functools.html#functools.partial
        - https://github.com/python/typeshed/blob/bbd9dd1c4f596f564542d48bb05b2cc2e2a7a28d/stdlib/functools.pyi#L129
    """

    @wraps(func)
    def __wrapper(*func_args: Any, **func_kwargs: Any) -> R:
        # FIXME: https://github.com/python/typeshed/issues/8703
        return func(*(func_args + fixed_args), **(func_kwargs | fixed_kwargs))

    return __wrapper


def accepts_varkwargs(func: Callable[..., Any], /) -> bool:
    r"""Check if function accepts kwargs."""
    sig = inspect.signature(func)
    return any(p.kind is VAR_KEYWORD for p in sig.parameters.values())


def dataclass_args_kwargs(
    obj: Dataclass, /, *, ignore_parent_fields: bool = False
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    r"""Return positional and keyword arguments of a dataclass."""
    if not isinstance(obj, Dataclass):
        raise TypeError(f"Expected dataclass, got {type(obj)}")

    forbidden_keys: set[str] = set()
    if ignore_parent_fields:
        for parent in obj.__class__.__mro__[1:]:
            if is_dataclass(parent):
                forbidden_keys.update(parent.__dataclass_fields__)

    args = tuple(
        getattr(obj, key)
        for key, val in obj.__dataclass_fields__.items()
        if not val.kw_only and key not in forbidden_keys
    )
    kwargs = {
        key: getattr(obj, key)
        for key, val in obj.__dataclass_fields__.items()
        if val.kw_only and key not in forbidden_keys
    }

    return args, kwargs


def get_parameter_kind(s: str | Kind, /) -> set[Kind]:
    r"""Get parameter kind from string."""
    if isinstance(s, Kind):
        return {s}

    match s.lower():
        case "p" | "positional":
            return {POSITIONAL_ONLY, VAR_POSITIONAL}
        case "k" | "keyword":
            return {KEYWORD_ONLY, VAR_KEYWORD}
        case "v" | "var":
            return {VAR_POSITIONAL, VAR_KEYWORD}
        case "po" | "positional_only":
            return {POSITIONAL_ONLY}
        case "ko" | "keyword_only":
            return {KEYWORD_ONLY}
        case "pk" | "positional_or_keyword":
            return {POSITIONAL_OR_KEYWORD}
        case "vp" | "var_positional":
            return {VAR_POSITIONAL}
        case "vk" | "var_keyword":
            return {VAR_KEYWORD}
    raise ValueError(f"Unknown kind {s}")


def get_function_args(
    func: Callable[..., Any],
    /,
    *,
    mandatory: Optional[bool] = None,
    kinds: Optional[str | Kind | list[Kind]] = None,
) -> list[Parameter]:
    r"""Filter function parameters by kind and optionality."""
    match kinds:
        case None:
            allowed_kinds = set(Kind)
        case str() | Kind():
            allowed_kinds = get_parameter_kind(kinds)
        case Sequence():
            allowed_kinds = set().union(*map(get_parameter_kind, kinds))
        case _:
            raise ValueError(f"Unknown type for kinds: {type(kinds)}")

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if mandatory is None:
        return [p for p in params if p.kind in allowed_kinds]

    return [
        p
        for p in params
        if is_mandatory_arg(p) is mandatory and p.kind in allowed_kinds
    ]


def get_mandatory_argcount(func: Callable, /) -> int:
    r"""Get the number of mandatory arguments of a function."""
    sig = inspect.signature(func)
    return sum(is_mandatory_arg(p) for p in sig.parameters.values())


def get_mandatory_kwargs(func: Callable, /) -> set[str]:
    r"""Get the mandatory keyword arguments of a function."""
    sig = inspect.signature(func)
    return {
        name
        for name, p in sig.parameters.items()
        if is_mandatory_arg(p) and is_keyword_arg(p)
    }


def get_parameter(func: Callable, name: str, /) -> Parameter:
    """Get parameter from function."""
    sig = inspect.signature(func)
    if name not in sig.parameters:
        raise ValueError(f"{func=} takes np argument named {name!r}.")
    return sig.parameters[name]


def get_return_typehint(func: Callable, /) -> Any:
    r"""Get the return typehint of a function."""
    # if isinstance(self.func, FunctionType | MethodType):
    #     ann = self.func.__annotations__.get("return", object)  # type: ignore[unreachable]
    # else:
    #     ann = self.func.__call__.__annotations__.get("return", object)  # type: ignore[operator]
    #
    # ann.__name__ if isinstance(ann, type) else str(ann)
    sig = inspect.signature(func)
    ann = sig.return_annotation

    match ann:
        case type():
            return ann.__name__
        case sig.empty:
            return Any
        case _:
            return ann


@overload
def is_mandatory_arg(param: Parameter, /) -> bool: ...
@overload
def is_mandatory_arg(func: Callable, name: str, /) -> bool: ...
def is_mandatory_arg(func_or_param, name=None, /):
    r"""Check if parameter is mandatory."""
    match func_or_param, name:
        case Parameter() as param, None:
            return param.default is Parameter.empty and param.kind not in {
                VAR_POSITIONAL,
                VAR_KEYWORD,
            }
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)
            return is_mandatory_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_positional_arg(param: Parameter, /) -> bool: ...
@overload
def is_positional_arg(func: Callable, name: str, /) -> bool: ...
def is_positional_arg(func_or_param, name=None, /):
    r"""Check if parameter is positional argument."""
    match func_or_param, name:
        case Parameter() as param, None:
            return param.kind in {
                POSITIONAL_ONLY,
                POSITIONAL_OR_KEYWORD,
                VAR_POSITIONAL,
            }
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)
            return is_positional_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_positional_only_arg(param: Parameter, /) -> bool: ...
@overload
def is_positional_only_arg(func: Callable, name: str, /) -> bool: ...
def is_positional_only_arg(func_or_param, name=None, /):
    """Check if parameter is positional only argument."""
    match func_or_param, name:
        case Parameter() as param, None:
            return param.kind in {POSITIONAL_ONLY, VAR_POSITIONAL}
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)
            return is_positional_only_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_keyword_arg(param: Parameter, /) -> bool: ...
@overload
def is_keyword_arg(func: Callable, name: str, /) -> bool: ...
def is_keyword_arg(func_or_param, name=None, /):
    """Check if parameter is keyword argument."""
    match func_or_param, name:
        case Parameter() as param, None:
            return param.kind in {KEYWORD_ONLY, POSITIONAL_OR_KEYWORD, VAR_KEYWORD}
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)
            return is_keyword_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_keyword_only_arg(param: Parameter, /) -> bool: ...
@overload
def is_keyword_only_arg(func: Callable, name: str, /) -> bool: ...
def is_keyword_only_arg(func_or_param, name=None, /):
    """Check if parameter is keyword only argument."""
    match func_or_param, name:
        case Parameter() as param, None:
            return param.kind in {KEYWORD_ONLY, VAR_KEYWORD}
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)
            return is_keyword_only_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_variadic_arg(param: Parameter, /) -> bool: ...
@overload
def is_variadic_arg(func: Callable, name: str, /) -> bool: ...
def is_variadic_arg(func_or_param, name=None, /):
    """Check if parameter is variadic argument."""
    match func_or_param, name:
        case Parameter() as param, None:
            return param.kind in {VAR_POSITIONAL, VAR_KEYWORD}
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)
            return is_variadic_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


def prod_fn(*funcs: Callable) -> Callable:
    r"""Cartesian Product of Functions.

    It is assumed every function takes a single positional argument.
    """

    def __prod_fn(args, /):
        """Argument is a tuple with the input for each function."""
        return tuple(f(arg) for f, arg in zip(funcs, args, strict=True))

    return __prod_fn
