r"""Test namedtuple decorator."""

import ast
import textwrap
from collections.abc import Callable as Fn, Sequence
from functools import partial, wraps
from inspect import getsource
from typing import Any, NamedTuple, Optional, Protocol, overload


class Decorator[T_in, T_out, **P](Protocol):
    r"""Protocol for decorators."""

    def __call__(self, obj: T_in, /, *args: P.args, **kwargs: P.kwargs) -> T_out: ...


class ParametrizedDecorator[T_in, T_out, **P](Protocol):
    r"""Protocol for parametrized decorators."""

    @overload  # @decorator(*args, **kwargs)
    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[T_in], T_out]: ...
    @overload  # @decorator / decorator(obj, *args, **kwargs)
    def __call__(self, obj: T_in, /, *args: P.args, **kwargs: P.kwargs) -> T_out: ...


class PolymorphicDecorator[**P](Protocol):
    r"""Polymorphic Decorator Protocol."""

    @overload  # @decorator
    def __call__[T](self, obj: T, /, *args: P.args, **kwargs: P.kwargs) -> T: ...
    @overload  # @decorator(*args, **kwargs)
    def __call__[T](self, /, *args: P.args, **kwargs: P.kwargs) -> Fn[[T], T]: ...


def decorator[X, Y, **P](deco: Decorator[X, Y, P], /) -> ParametrizedDecorator[X, Y, P]:
    r"""Decorator Factory."""
    OBJ: Any = object()

    @wraps(deco)
    def _deco(obj: X = OBJ, /, *args: P.args, **kwargs: P.kwargs) -> Y | Fn[[X], Y]:
        if obj is OBJ:
            return partial(deco, *args, **kwargs)
        return deco(obj, *args, **kwargs)

    return _deco  # type: ignore[return-value]


def get_exit_point_names(func: Fn, /) -> list[tuple[str, ...]]:
    r"""Return the variable names used in exit nodes."""
    source = textwrap.dedent(getsource(func))
    tree = ast.parse(source)
    exit_points = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]

    var_names = []
    for exit_point in exit_points:
        assert isinstance(exit_point.value, ast.Tuple)

        e: tuple[str, ...] = ()
        for obj in exit_point.value.elts:
            assert isinstance(obj, ast.Name)
            e += (obj.id,)
        var_names.append(e)
    return var_names


@decorator  # type: ignore[arg-type]
def return_namedtuple[**P](
    func: Fn[P, tuple],
    /,
    *,
    name: Optional[str] = None,
    field_names: Optional[Sequence[str]] = None,
) -> Fn[P, tuple]:
    r"""Convert a function's return type to a namedtuple."""
    name = f"{func.__name__}_tuple" if name is None else name

    # noinspection PyUnresolvedReferences
    return_type = func.__annotations__.get("return", NotImplemented)
    if return_type is NotImplemented:
        raise ValueError("No return type hint found.")
    if not issubclass(return_type.__origin__, tuple):
        raise TypeError("Return type hint is not a tuple.")

    type_hints = return_type.__args__
    potential_return_names = set(get_exit_point_names(func))

    if len(type_hints) == 0:
        raise ValueError("Return type hint is an empty tuple.")
    if Ellipsis in type_hints:
        raise ValueError("Return type hint is a variable length tuple.")
    if field_names is None:
        if len(potential_return_names) != 1:
            raise ValueError("Automatic detection of names failed.")
        field_names = potential_return_names.pop()
    elif any(len(r) != len(type_hints) for r in potential_return_names):
        raise ValueError("Number of names does not match number of return values.")

    # create namedtuple
    tuple_type: type[tuple] = NamedTuple(  # type: ignore[misc]
        name, zip(field_names, type_hints, strict=True)
    )

    @wraps(func)
    def _wrapper(*func_args: P.args, **func_kwargs: P.kwargs) -> tuple:
        # noinspection PyCallingNonCallable
        return tuple_type(*func(*func_args, **func_kwargs))

    return _wrapper


def test_namedtuple_decorator() -> None:
    @return_namedtuple  # type: ignore[arg-type]
    def foo(x: int, y: int) -> tuple[int, int]:
        q, r = divmod(x, y)
        return q, r

    assert str(foo(5, 3)) == "foo_tuple(q=1, r=2)"

    @return_namedtuple(name="divmod")  # type: ignore[arg-type]
    def bar(x: int, y: int) -> tuple[int, int]:
        q, r = divmod(x, y)
        return q, r

    assert str(bar(5, 3)) == "divmod(q=1, r=2)"
