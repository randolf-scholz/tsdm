from functools import partial, wraps
from typing import Any, Callable, Protocol, TypeVar, assert_type, overload, reveal_type

T = TypeVar("T")
C = TypeVar("C", bound=Callable)


class ClassDecorator(Protocol[T]):
    def __call__(self, cls: type[T], /) -> type[T]: ...


class ClassDecoratorFactory(Protocol[T]):
    def __call__(self, /, *args: Any, **kwargs: Any) -> ClassDecorator[T]: ...


class ParametrizedClassDecorator(Protocol[T]):
    @overload
    def __call__(self, /, **kwargs: Any) -> ClassDecorator[T]: ...
    @overload
    def __call__(self, cls: type[T], /, **kwargs: Any) -> type[T]: ...


class FunctionDecorator(Protocol[C]):
    def __call__(self, func: C, /) -> C: ...


class FunctionDecoratorFactory(Protocol[C]):
    def __call__(self, /, *args: Any, **kwargs: Any) -> FunctionDecorator[C]: ...


class ParametrizedFunctionDecorator(Protocol[C]):
    @overload
    def __call__(self, /, **kwargs: Any) -> FunctionDecorator[C]: ...
    @overload
    def __call__(self, func: C, /, **kwargs: Any) -> C: ...


class Decorator(Protocol[T]):
    def __call__(self, obj: T, /) -> T: ...


class DecoratorFactory(Protocol[T]):
    def __call__(self, /, *args: Any, **kwargs: Any) -> Decorator[T]: ...


class ParametrizedDecorator(Protocol[T]):
    @overload
    def __call__(self, /, **kwargs: Any) -> Decorator[T]: ...
    @overload
    def __call__(self, obj: T, /, **kwargs: Any) -> T: ...


@overload
def decorator(deco: ClassDecorator[T], /) -> ParametrizedClassDecorator[T]: ...
@overload
def decorator(deco: FunctionDecorator[C], /) -> ParametrizedFunctionDecorator[C]: ...
def decorator(deco, /):
    @wraps(deco)
    def __parametrized_decorator(obj=None, /, **kwargs):
        if obj is None:
            return partial(deco, **kwargs)
        return deco(obj, **kwargs)  # type: ignore

    return __parametrized_decorator  # type: ignore


@decorator
def pprint(cls: type[T], /, **options: Any) -> type[T]:
    r"""Add custom __repr__ to class."""
    cls.__repr__ = partial(repr, cls)  # type: ignore
    return cls


@decorator
def trace(func: C, /, **options: Any) -> C:
    @wraps(func)
    def __trace(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args} and kwargs={kwargs}")
        return func(*args, **kwargs)

    return __trace  # type: ignore


assert_type(pprint, ParametrizedClassDecorator)
assert_type(trace, ParametrizedFunctionDecorator)


@pprint
class A: ...


@pprint(indent=4)
class B: ...


assert_type(A, type[A])
assert_type(B, type[B])


@trace
def f(x: int) -> int:
    return x + 1


@trace(indent=4)
def g(x: int) -> int:
    return x + 1


assert_type(f, Callable[[int], int])
assert_type(g, Callable[[int], int])
