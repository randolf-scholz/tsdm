r"""Experimental decorators."""

__all__ = [
    "attribute",
    "abstractattribute",
    "DummyAttribute",
    "PatchedABCMeta",
    "return_namedtuple",
    "get_exit_point_names",
]

import ast
import textwrap
from abc import ABCMeta
from collections.abc import Callable as Fn, Sequence
from dataclasses import dataclass, field
from functools import wraps
from inspect import getsource
from typing import (
    Any,
    ClassVar,
    NamedTuple,
    Optional,
    Self,
    cast,
    overload,
)

from tsdm.utils.decorators.base import DecoratorError, decorator


class _AttrMeta(type):
    r"""Metaclass for attribute decorators."""

    def __call__[T, R](cls, func: Fn[[T], R], /) -> R:
        r"""Create a decorator that converts method to attribute."""
        _attr = super().__call__(func)
        wrapper = wraps(func, updated=())
        attr = cast(R, wrapper(_attr))
        return attr


@dataclass
class attribute[T, R](metaclass=_AttrMeta):
    r"""Create a decorator that converts method to attribute."""

    SENTINEL: ClassVar[Any] = object()
    DELETED: ClassVar[Any] = object()

    func: Fn[[T], R]
    payload: R = field(default=SENTINEL, init=False)

    @overload
    def __get__(self, obj: None, obj_type: Optional[type] = ..., /) -> Self: ...
    @overload
    def __get__(self, obj: T, obj_type: Optional[type] = ..., /) -> R: ...
    def __get__(self, obj: None | T, obj_type: Optional[type] = None) -> Self | R:
        if obj is None:
            return self
        if self.payload is self.DELETED:
            raise AttributeError("Attribute has been deleted.")
        if self.payload is self.SENTINEL:
            self.payload = self.func(obj)
        return self.payload

    def __set__(self, obj: T, value: R, /) -> None:
        self.payload = value

    def __delete__(self, obj: T, /) -> None:
        self.payload = self.DELETED


class DummyAttribute:
    r"""Sentinel for abstract attributes."""

    __is_abstract_attribute__ = True


def abstractattribute[R](obj: Optional[Fn[[Any], R]] = None) -> R:
    r"""Decorate method as abstract attribute."""
    attr = DummyAttribute() if obj is None else obj
    try:
        attr.__is_abstract_attribute__ = True  # type: ignore[union-attr]
    except AttributeError as exc:
        raise AttributeError(
            f"Cannot decorate with abstractattribute decorator because {obj} "
            "does not support setting attributes."
        ) from exc
    return cast(R, attr)


class PatchedABCMeta(ABCMeta):
    r"""Patched ABCMeta class to allow @abstractattribute."""

    def __call__(cls, *args: Any, **kwargs: Any) -> "PatchedABCMeta":
        r"""Override __call__ to allow @abstractattribute."""
        instance: PatchedABCMeta = ABCMeta.__call__(cls, *args, **kwargs)
        abstract_attributes = {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), "__is_abstract_attribute__", False)
        }
        if abstract_attributes:
            raise NotImplementedError(
                f"Can't instantiate abstract class {cls.__name__} with"
                f" abstract attributes: f{", ".join(abstract_attributes)}"
            )
        return instance


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


@decorator  # type: ignore[arg-type]
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
    tuple_type: type[tuple] = NamedTuple(  # type: ignore[misc]
        name, zip(field_names, type_hints, strict=True)
    )

    @wraps(func)
    def _wrapper(*func_args: P.args, **func_kwargs: P.kwargs) -> tuple:
        # noinspection PyCallingNonCallable
        return tuple_type(*func(*func_args, **func_kwargs))

    return _wrapper
