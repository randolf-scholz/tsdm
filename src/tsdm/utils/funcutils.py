r"""Utility functions relating to function arguments and return values."""

__all__ = [
    "KEYWORD_ONLY",
    "POSITIONAL_ONLY",
    "POSITIONAL_OR_KEYWORD",
    "VAR_KEYWORD",
    "VAR_POSITIONAL",
    # Types
    "TaggedCallable",
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
    "recurse_on_nested_builtin",
    "yield_return_nodes",
    "tags",
]

import ast
import inspect
from ast import AST, Name, Return, Tuple
from collections.abc import Callable as Fn, Iterable, Iterator, Sequence
from dataclasses import fields
from inspect import Parameter, _ParameterKind as ParameterKind, getsource
from typing import Any, Final, Optional, Protocol, cast, overload

from tsdm.types.dataclass import DataclassType, is_dataclass
from tsdm.types.nested import NestedBuiltin

KEYWORD_ONLY: Final = Parameter.KEYWORD_ONLY
POSITIONAL_ONLY: Final = Parameter.POSITIONAL_ONLY
POSITIONAL_OR_KEYWORD: Final = Parameter.POSITIONAL_OR_KEYWORD
VAR_KEYWORD: Final = Parameter.VAR_KEYWORD
VAR_POSITIONAL: Final = Parameter.VAR_POSITIONAL


class TaggedCallable[**P, R](Protocol):
    r"""Protocol for callables with a `__tags__` dictionary."""

    __tags__: dict[str, Any]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


class _MapsToTagged(Protocol):
    def __call__[**P, R](self, fn: Fn[P, R], /) -> TaggedCallable[P, R]: ...


def tags(**kwargs: object) -> _MapsToTagged:
    def decorator[**P, R](fn: Fn[P, R], /) -> TaggedCallable[P, R]:
        fn = cast("TaggedCallable[P, R]", fn)
        fn.__tags__ = kwargs
        return fn

    return decorator


def accepts_varkwargs(func: Fn[..., Any], /) -> bool:
    r"""Check whether a function accepts arbitrary keyword arguments."""
    sig = inspect.signature(func)
    return any(p.kind is VAR_KEYWORD for p in sig.parameters.values())


def dataclass_args_kwargs(
    obj: DataclassType, /, *, ignore_parent_fields: bool = False
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    r"""Return positional and keyword arguments of a dataclass."""
    forbidden_keys: set[str] = set()
    if ignore_parent_fields:
        for parent in obj.__class__.__mro__[1:]:
            if isinstance(parent, type) and is_dataclass(parent):
                forbidden_keys.update(field.name for field in fields(parent))

    args = tuple(
        getattr(obj, field.name)
        for field in fields(obj)
        if not field.kw_only and field.name not in forbidden_keys
    )
    kwargs = {
        field.name: getattr(obj, field.name)
        for field in fields(obj)
        if field.kw_only and field.name not in forbidden_keys
    }
    return args, kwargs


def get_parameter_kind(s: str | ParameterKind, /) -> set[ParameterKind]:
    r"""Return the parameter kinds selected by a kind or string alias."""
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
        case _:
            raise ValueError(f"Unknown kind {s}")


def get_function_args(
    func: Fn[..., Any],
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


def get_mandatory_argcount(func: Fn, /) -> int:
    r"""Get the number of mandatory arguments of a function."""
    sig = inspect.signature(func)
    return sum(is_mandatory_arg(p) for p in sig.parameters.values())


def get_mandatory_kwargs(func: Fn, /) -> set[str]:
    r"""Get the mandatory keyword arguments of a function."""
    sig = inspect.signature(func)
    return {
        name
        for name, p in sig.parameters.items()
        if is_mandatory_arg(p) and is_keyword_arg(p)
    }


def get_parameter(func: Fn, name: str, /) -> Parameter:
    r"""Get a named parameter from a function's signature."""
    sig = inspect.signature(func)
    if name not in sig.parameters:
        raise ValueError(f"{func=} takes np argument named {name!r}.")
    return sig.parameters[name]


def yield_return_nodes(nodes: Iterable[AST], /) -> Iterator[Return]:
    r"""Yield return statements from an iterable of AST nodes."""
    for node in nodes:
        if isinstance(node, Return):
            yield node


def _yield_names(nodes: Iterable[AST], /) -> Iterator[str]:
    r"""Yield identifiers from AST name nodes."""
    for obj in nodes:
        match obj:
            case Name(id=name):
                yield name
            case _:
                raise TypeError(f"Expected ast.Name, got {type(obj)}.")


def get_exit_point_names(func: Fn, /) -> set[tuple[str, ...]]:
    r"""Return the variable names in tuple-valued return statements.

    Each return value must be a tuple of names. Other return values raise
    `TypeError`.
    """
    tree = ast.parse(getsource(func))
    iter_nodes = ast.walk(tree)

    names: set[tuple[str, ...]] = set()
    for exit_point in yield_return_nodes(iter_nodes):
        match exit_point:
            case Return(value=Tuple(elts=elts)):
                names.add(tuple(_yield_names(elts)))
            case _:
                raise TypeError("Return value must be a tuple.")
    return names


def get_return_typehint(func: Fn, /) -> Any:
    r"""Return a function's return annotation for display.

    Return `"Unknown"` when no annotation is present and the type's name when
    the annotation is a type. Other annotations are returned unchanged.
    """
    sig = inspect.signature(func)
    ann = sig.return_annotation

    match ann:
        case sig.empty:
            return "Unknown"
        case type():
            return ann.__name__
        case _:
            return ann


@overload
def is_mandatory_arg(param: Parameter, /) -> bool: ...
@overload
def is_mandatory_arg(func: Fn, name: str, /) -> bool: ...
def is_mandatory_arg(arg: Fn | Parameter, name: Optional[str] = None, /) -> bool:
    r"""Check whether a parameter is required and nonvariadic."""
    match arg, name:
        case Parameter(kind=kind, default=default), None:
            return default is Parameter.empty and kind not in {
                VAR_POSITIONAL,
                VAR_KEYWORD,
            }
        case (fn, str(name)) if callable(fn):
            param = get_parameter(fn, name)
            return is_mandatory_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_positional_arg(param: Parameter, /) -> bool: ...
@overload
def is_positional_arg(func: Fn, name: str, /) -> bool: ...
def is_positional_arg(arg: Fn | Parameter, name: Optional[str] = None, /) -> bool:
    r"""Check whether a parameter accepts positional arguments."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {
                POSITIONAL_ONLY,
                POSITIONAL_OR_KEYWORD,
                VAR_POSITIONAL,
            }
        case (fn, str(name)) if callable(fn):
            param = get_parameter(fn, name)
            return is_positional_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_positional_only_arg(param: Parameter, /) -> bool: ...
@overload
def is_positional_only_arg(func: Fn, name: str, /) -> bool: ...
def is_positional_only_arg(arg: Parameter | Fn, name: Optional[str] = None, /) -> bool:
    r"""Check whether a parameter accepts only positional arguments."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {POSITIONAL_ONLY, VAR_POSITIONAL}
        case (fn, str(name)) if callable(fn):
            param = get_parameter(fn, name)
            return is_positional_only_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_keyword_arg(param: Parameter, /) -> bool: ...
@overload
def is_keyword_arg(func: Fn, name: str, /) -> bool: ...
def is_keyword_arg(arg: Parameter | Fn, name: Optional[str] = None, /) -> bool:
    r"""Check whether a parameter accepts keyword arguments."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {
                KEYWORD_ONLY,
                POSITIONAL_OR_KEYWORD,
                VAR_KEYWORD,
            }
        case (fn, str(name)) if callable(fn):
            param = get_parameter(fn, name)
            return is_keyword_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_keyword_only_arg(param: Parameter, /) -> bool: ...
@overload
def is_keyword_only_arg(func: Fn, name: str, /) -> bool: ...
def is_keyword_only_arg(arg: Parameter | Fn, name: Optional[str] = None, /) -> bool:
    r"""Check whether a parameter accepts only keyword arguments."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {KEYWORD_ONLY, VAR_KEYWORD}
        case (fn, str(name)) if callable(fn):
            param = get_parameter(fn, name)
            return is_keyword_only_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


@overload
def is_variadic_arg(param: Parameter, /) -> bool: ...
@overload
def is_variadic_arg(func: Fn, name: str, /) -> bool: ...
def is_variadic_arg(arg: Parameter | Fn, name: Optional[str] = None, /) -> bool:
    r"""Check whether a parameter accepts variadic arguments."""
    match arg, name:
        case Parameter(kind=kind), None:
            return kind in {VAR_POSITIONAL, VAR_KEYWORD}
        case (fn, str(name)) if callable(fn):
            param = get_parameter(fn, name)
            return is_variadic_arg(param)
        case _:
            raise TypeError("Unsupported input types.")


def prod_fn(*funcs: Fn[[Any], Any]) -> Fn[[tuple], tuple]:
    r"""Return a function that applies each supplied function to its matching input.

    Each function must accept a single positional argument. The returned function
    takes a tuple with one input per function and returns a tuple of results.
    A mismatched number of inputs raises `ValueError`.
    """

    def __prod_fn(args: tuple, /) -> tuple:
        r"""Apply each function to the corresponding item in the input tuple."""
        return tuple(f(arg) for f, arg in zip(funcs, args, strict=True))

    return __prod_fn


def recurse_on_nested_builtin[T, R](
    arg: NestedBuiltin[T],
    /,
    *,
    leaf_fn: Fn[[T], R],
    leaf_type: type[T],
    # optional arguments
    leaf_priotizied: bool = False,
) -> NestedBuiltin[R]:
    r"""Recursively apply a function to the leaves of a nested data structure.

    Args:
        arg: Nested data structure to apply the function to.
        leaf_fn: Function to apply to the leaf nodes.
        leaf_type: Type of the leaf nodes.
        leaf_priotizied: Whether to check the leaf type before container types.
            If false, supported containers are traversed even if they match
            `leaf_type`.

    Returns:
        A structure with the same container types and transformed leaf values.

    Raises:
        TypeError: If a value is neither a supported builtin container nor a leaf.
    """

    def recurse(obj: NestedBuiltin[T], /) -> NestedBuiltin[R]:
        match obj:
            case leaf if leaf_priotizied and isinstance(leaf, leaf_type):
                return leaf_fn(leaf)
            case dict(mapping):
                return {k: recurse(v) for k, v in mapping.items()}
            case list(seq):
                return [recurse(obj) for obj in seq]
            case tuple(seq):
                return tuple(recurse(obj) for obj in seq)
            case frozenset(items):
                return frozenset(recurse(obj) for obj in items)
            case set(items):
                # FIXME: https://github.com/python/typeshed/issues/9571
                return {recurse(obj) for obj in items}  # pyright: ignore[reportUnhashable]
            case leaf if isinstance(leaf, leaf_type):
                return leaf_fn(leaf)
            case _:
                raise TypeError(
                    f"Unsupported type: {type(obj)} not an instance of {leaf_type} or one of the builtin containers"
                    f" {dict, list, tuple, frozenset, set}."
                )

    return recurse(arg)
