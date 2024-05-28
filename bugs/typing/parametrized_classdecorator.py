# fmt: off
from functools import partial, partialmethod, wraps
from typing import Any, Protocol, TypeVar, assert_type, overload

T = TypeVar("T")

class ClassDecorator(Protocol[T]):
    def __call__(self, cls: type[T], /) -> type[T]: ...

class ParametrizedClassDecorator(Protocol[T]):
    @overload
    def __call__(self, /, **kwargs: Any) -> ClassDecorator[T]: ...
    @overload
    def __call__(self, cls: type[T], /, **kwargs: Any) -> type[T]: ...

def decorator(deco: ClassDecorator[T], /) -> ParametrizedClassDecorator[T]:
    @wraps(deco)
    def __parametrized_decorator(obj=None, /, **kwargs):
        if obj is None:
            return partial(deco, **kwargs)
        return deco(obj, **kwargs)
    return __parametrized_decorator  # pyright: ignore[reportReturnType]

def format_obj(obj: object, **options: Any) -> str:
    return f"Custom formatting of {object.__repr__(obj)}"  # dummy implementation

@decorator
def pprint(cls: type[T], /, **options: Any) -> type[T]:
    cls.__repr__ = partialmethod(format_obj, **options)  # type: ignore
    return cls

# Test cases
@pprint  # ❌ type[A] cannot be assigned to type[T@pprint]
class A: ...
@pprint(indent=4)  # ❌ type[B] cannot be assigned to type[T@pprint]
class B: ...
class C: ...
class D: ...

assert_type(A, type[A])
assert_type(B, type[B])
assert_type(pprint(C), type[C])  # ❌ type[C] cannot be assigned to type[T@pprint]
assert_type(pprint(D, indent=4), type[D])  # ❌ type[D] cannot be assigned to type[T@pprint]
print(A())
