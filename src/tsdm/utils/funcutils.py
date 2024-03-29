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

from typing_extensions import Any, Optional, ParamSpec, cast, overload

from tsdm.types.protocols import Dataclass, is_dataclass
from tsdm.types.variables import return_var_co as R

KEYWORD_ONLY = Parameter.KEYWORD_ONLY
POSITIONAL_ONLY = Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD = Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD = Parameter.VAR_KEYWORD
VAR_POSITIONAL = Parameter.VAR_POSITIONAL
Kind = inspect._ParameterKind  # pylint: disable=protected-access

P = ParamSpec("P")


def accepts_varkwargs(f: Callable[..., Any]) -> bool:
    r"""Check if function accepts kwargs."""
    sig = inspect.signature(f)
    return any(p.kind is VAR_KEYWORD for p in sig.parameters.values())


def dataclass_args_kwargs(
    obj: Dataclass, *, ignore_parent_fields: bool = False
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
    f: Callable[..., Any],
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

    sig = inspect.signature(f)
    params = list(sig.parameters.values())

    if mandatory is None:
        return [p for p in params if p.kind in allowed_kinds]

    return [
        p
        for p in params
        if is_mandatory_arg(p) is mandatory and p.kind in allowed_kinds
    ]


def get_mandatory_argcount(f: Callable[..., Any]) -> int:
    r"""Get the number of mandatory arguments of a function."""
    sig = inspect.signature(f)
    return sum(is_mandatory_arg(p) for p in sig.parameters.values())


def get_mandatory_kwargs(f: Callable[..., Any]) -> set[str]:
    r"""Get the mandatory keyword arguments of a function."""
    sig = inspect.signature(f)
    return {
        name
        for name, p in sig.parameters.items()
        if is_mandatory_arg(p) and is_keyword_arg(p)
    }


def get_return_typehint(f: Callable[..., Any]) -> Any:
    r"""Get the return typehint of a function."""
    # if isinstance(self.func, FunctionType | MethodType):
    #     ann = self.func.__annotations__.get("return", object)  # type: ignore[unreachable]
    # else:
    #     ann = self.func.__call__.__annotations__.get("return", object)  # type: ignore[operator]
    #
    # ann.__name__ if isinstance(ann, type) else str(ann)
    sig = inspect.signature(f)
    ann = sig.return_annotation

    match ann:
        case type():
            return ann.__name__
        case sig.empty:
            return Any
        case _:
            return ann


def is_mandatory_arg(p: Parameter, /) -> bool:
    r"""Check if parameter is mandatory."""
    return p.default is Parameter.empty and p.kind not in (
        VAR_POSITIONAL,
        VAR_KEYWORD,
    )


@overload
def is_positional_arg(p: Parameter, /) -> bool: ...
@overload
def is_positional_arg(p: str, /, *, func: Callable) -> bool: ...
def is_positional_arg(p, /, *, func=None):
    r"""Check if parameter is positional argument."""
    match p, func:
        case Parameter() as param, None:
            return param.kind in (
                POSITIONAL_ONLY,
                POSITIONAL_OR_KEYWORD,
                VAR_POSITIONAL,
            )
        case str() as name, Callable() as function:  # type: ignore[misc]
            # FIXME: https://github.com/python/cpython/issues/102395
            function = cast(Callable, function)  # type: ignore[has-type]
            sig = inspect.signature(function)
            if name not in sig.parameters:
                raise ValueError(
                    f"Function {function} takes np argument named {name!r}."
                )
            return is_positional_arg(sig.parameters[name])
        case _:
            raise TypeError("Unsupported input types.")


def is_positional_only_arg(p: Parameter, /) -> bool:
    """Check if parameter is positional only argument."""
    return p.kind in (POSITIONAL_ONLY, VAR_POSITIONAL)


def is_keyword_only_arg(p: Parameter, /) -> bool:
    """Check if parameter is keyword only argument."""
    return p.kind in (KEYWORD_ONLY, VAR_KEYWORD)


def is_keyword_arg(p: Parameter, /) -> bool:
    """Check if parameter is keyword argument."""
    return p.kind in (POSITIONAL_OR_KEYWORD, KEYWORD_ONLY, VAR_KEYWORD)


def is_variadic_arg(p: Parameter, /) -> bool:
    """Check if parameter is variadic argument."""
    return p.kind in (VAR_POSITIONAL, VAR_KEYWORD)


def rpartial(
    func: Callable[P, R], /, *fixed_args: Any, **fixed_kwargs: Any
) -> Callable[..., R]:
    r"""Apply positional arguments from the right.

    References:
        - <https://docs.python.org/3/library/functools.html#functools.partial>
        - <https://github.com/python/typeshed/blob/bbd9dd1c4f596f564542d48bb05b2cc2e2a7a28d/stdlib/functools.pyi#L129>
    """

    @wraps(func)
    def _wrapper(*func_args: Any, **func_kwargs: Any) -> R:
        # TODO: https://github.com/python/typeshed/issues/8703
        return func(*(func_args + fixed_args), **(func_kwargs | fixed_kwargs))

    return _wrapper
