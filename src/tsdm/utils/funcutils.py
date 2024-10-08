r"""Utility functions relating to function arguments and return values."""

__all__ = [
    # Functions
    "accepts_varkwargs",
    "dataclass_args_kwargs",
    "get_exit_point_names",
    "get_function_args",
    "get_mandatory_argcount",
    "get_mandatory_kwargs",
    "get_parameter",
    "get_parameter_kind",
    "get_return_typehint",
    "is_keyword_arg",
    "is_keyword_only_arg",
    "is_mandatory_arg",
    "is_positional_arg",
    "is_positional_only_arg",
    "is_variadic_arg",
    "prod_fn",
    "rpartial",
    "yield_return_nodes",
]

import ast
import inspect
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import wraps
from inspect import Parameter, _ParameterKind as ParameterKind, getsource
from typing import Any, Optional, overload

from tsdm.constants import (
    KEYWORD_ONLY,
    POSITIONAL_ONLY,
    POSITIONAL_OR_KEYWORD,
    VAR_KEYWORD,
    VAR_POSITIONAL,
)
from tsdm.types.protocols import Dataclass, issubclass_dataclass


def rpartial[**P, R](  # +R
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


# FIXME: do not use __dataclass_fields__! Use dataclasses.fields instead.
def dataclass_args_kwargs(
    obj: Dataclass, /, *, ignore_parent_fields: bool = False
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    r"""Return positional and keyword arguments of a dataclass."""
    if not isinstance(obj, Dataclass):
        raise TypeError(f"Expected dataclass, got {type(obj)}")

    forbidden_keys: set[str] = set()
    if ignore_parent_fields:
        for parent in obj.__class__.__mro__[1:]:
            if issubclass_dataclass(parent):
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


def get_parameter_kind(s: str | ParameterKind, /) -> set[ParameterKind]:
    r"""Get parameter kind from string."""
    if isinstance(s, ParameterKind):
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
    kinds: Optional[str | ParameterKind | list[ParameterKind]] = None,
) -> list[Parameter]:
    r"""Filter function parameters by kind and optionality."""
    match kinds:
        case None:
            allowed_kinds = set(ParameterKind)
        case str() | ParameterKind():
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
    r"""Get parameter from function."""
    sig = inspect.signature(func)
    if name not in sig.parameters:
        raise ValueError(f"{func=} takes np argument named {name!r}.")
    return sig.parameters[name]


def yield_return_nodes(nodes: Iterable[ast.AST], /) -> Iterator[ast.Return]:
    r"""Collect all exit points of a function as ast nodes."""
    for node in nodes:
        match node:
            case ast.Return():
                yield node


def _yield_names(nodes: Iterable[ast.AST], /) -> Iterator[str]:
    r"""Yield variable names from ast nodes."""
    for obj in nodes:
        match obj:
            case ast.Name(id=name):
                yield name
            case _:
                raise TypeError(f"Expected ast.Name, got {type(obj)}.")


def get_exit_point_names(func: Callable, /) -> set[tuple[str, ...]]:
    r"""Return the variable names used in exit nodes."""
    tree = ast.parse(getsource(func))
    iter_nodes = ast.walk(tree)

    names: set[tuple[str, ...]] = set()
    for exit_point in yield_return_nodes(iter_nodes):
        match exit_point:
            case ast.Return(value=ast.Tuple(elts=elts)):
                names.add(tuple(_yield_names(elts)))
            case _:
                raise TypeError("Return value must be a tuple.")
    return names


def get_return_typehint(func: Callable, /) -> Any:
    r"""Get the return typehint of a function."""
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
def is_mandatory_arg(arg: Callable | Parameter, name: Optional[str] = None, /) -> bool:
    r"""Check if parameter is mandatory."""
    match arg, name:
        case Parameter(kind=kind, default=default), None:
            return default is Parameter.empty and kind not in {
                VAR_POSITIONAL,
                VAR_KEYWORD,
            }
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)  # type: ignore[unreachable]
            return is_mandatory_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_positional_arg(param: Parameter, /) -> bool: ...
@overload
def is_positional_arg(func: Callable, name: str, /) -> bool: ...
def is_positional_arg(arg: Callable | Parameter, name: Optional[str] = None, /) -> bool:
    r"""Check if parameter is positional argument."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {
                POSITIONAL_ONLY,
                POSITIONAL_OR_KEYWORD,
                VAR_POSITIONAL,
            }
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)  # type: ignore[unreachable]
            return is_positional_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_positional_only_arg(param: Parameter, /) -> bool: ...
@overload
def is_positional_only_arg(func: Callable, name: str, /) -> bool: ...
def is_positional_only_arg(
    arg: Parameter | Callable, name: Optional[str] = None, /
) -> bool:
    r"""Check if parameter is positional only argument."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {POSITIONAL_ONLY, VAR_POSITIONAL}
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)  # type: ignore[unreachable]
            return is_positional_only_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_keyword_arg(param: Parameter, /) -> bool: ...
@overload
def is_keyword_arg(func: Callable, name: str, /) -> bool: ...
def is_keyword_arg(arg: Parameter | Callable, name: Optional[str] = None, /) -> bool:
    r"""Check if parameter is keyword argument."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {
                KEYWORD_ONLY,
                POSITIONAL_OR_KEYWORD,
                VAR_KEYWORD,
            }
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)  # type: ignore[unreachable]
            return is_keyword_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_keyword_only_arg(param: Parameter, /) -> bool: ...
@overload
def is_keyword_only_arg(func: Callable, name: str, /) -> bool: ...
def is_keyword_only_arg(
    arg: Parameter | Callable, name: Optional[str] = None, /
) -> bool:
    r"""Check if parameter is keyword-only argument."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {KEYWORD_ONLY, VAR_KEYWORD}
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)  # type: ignore[unreachable]
            return is_keyword_only_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_variadic_arg(param: Parameter, /) -> bool: ...
@overload
def is_variadic_arg(func: Callable, name: str, /) -> bool: ...
def is_variadic_arg(arg: Parameter | Callable, name: Optional[str] = None, /) -> bool:
    r"""Check if parameter is variadic argument."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {VAR_POSITIONAL, VAR_KEYWORD}
        case Callable() as function, str(name):  # type: ignore[misc]
            param = get_parameter(function, name)  # type: ignore[unreachable]
            return is_variadic_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


def prod_fn(*funcs: Callable[[Any], Any]) -> Callable[[tuple], tuple]:
    r"""Cartesian Product of Functions.

    It is assumed every function takes a single positional argument.
    """

    def __prod_fn(args: tuple, /) -> tuple:
        r"""Argument is a tuple with the input for each function."""
        return tuple(f(arg) for f, arg in zip(funcs, args, strict=True))

    return __prod_fn
