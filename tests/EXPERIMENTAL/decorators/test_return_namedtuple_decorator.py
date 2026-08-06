import ast
import textwrap
from collections.abc import Callable as Fn, Sequence
from functools import wraps
from inspect import getsource
from typing import NamedTuple, Optional

from tsdm.decorator import DecoratorError, decorator


def get_exit_point_names(func: Fn, /) -> list[tuple[str, ...]]:
    r"""Return the variable names used in exit nodes."""
    source = textwrap.dedent(getsource(func))
    tree = ast.parse(source)
    exit_points = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]

    var_names = []
    for exit_point in exit_points:
        if not isinstance(exit_point.value, ast.Tuple):
            raise TypeError("Return value must be a tuple.")

        e: tuple[str, ...] = ()
        for obj in exit_point.value.elts:
            if not isinstance(obj, ast.Name):
                raise TypeError("Return value must be a tuple of variables.")
            e += (obj.id,)
        var_names.append(e)
    return var_names


@decorator  # type: ignore
def return_namedtuple[**P](
    func: Fn[P, tuple],
    /,
    *,
    name: Optional[str] = None,
    field_names: Optional[Sequence[str]] = None,
) -> Fn[P, tuple]:
    r"""Convert a function's return type to a namedtuple."""
    # noinspection PyUnresolvedReferences
    annotations: dict = func.__annotations__
    name = f"{func.__name__}_tuple" if name is None else name
    if "return" not in annotations:
        raise DecoratorError(func, "No return type hint found.")
    return_type = annotations["return"]

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
    tuple_type: type[tuple] = NamedTuple(  # pyrefly: ignore[bad-assignment]
        # FIXME: https://github.com/python/cpython/issues/144321
        name,
        list(zip(field_names, type_hints, strict=True)),  # pyrefly: ignore[bad-argument-count]
    )

    @wraps(func)
    def _wrapper(*func_args: P.args, **func_kwargs: P.kwargs) -> tuple:
        # noinspection PyCallingNonCallable
        return tuple_type(*func(*func_args, **func_kwargs))

    return _wrapper


def test_namedtuple_decorator() -> None:
    @return_namedtuple
    def foo(x: int, y: int) -> tuple[int, int]:
        q, r = divmod(x, y)
        return q, r

    assert str(foo(5, 3)) == "foo_tuple(q=1, r=2)"

    @return_namedtuple(name="divmod")
    def bar(x: int, y: int) -> tuple[int, int]:
        q, r = divmod(x, y)
        return q, r

    assert str(bar(5, 3)) == "divmod(q=1, r=2)"
