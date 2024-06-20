from functools import partialmethod
from typing import Any, Callable, Concatenate, TypeVar, reveal_type

Cls = TypeVar("Cls", bound=type)
T = TypeVar("T")


def my_pprint(obj: Any, /, **options: Any) -> str:
    return "..."


# repr_fn = my_pprint  # 4 issues
repr_fn: Callable[..., str] = my_pprint  # 2 issues
# repr_fn: Callable[Concatenate[object, ...], str] = my_pprint  # 4 issues


# cls w/ partialmethod
def pprint1(cls: Cls, /, **kwds: Any) -> Cls:
    reveal_type(cls.__repr__)  # "() -> str"
    cls.__repr__ = partialmethod(repr_fn, **kwds)  # ❌ incompatible type
    return cls


# cls w/o partialmethod
def pprint2(cls: Cls, /) -> Cls:
    reveal_type(cls.__repr__)  # "() -> str"
    cls.__repr__ = repr_fn  # ✅
    return cls


# type[T] w/ partialmethod
def pprint3(cls: type[T], /, **kwds: Any) -> type[T]:
    reveal_type(cls.__repr__)  # "(self: T@pprint3) -> str"
    cls.__repr__ = partialmethod(repr_fn, **kwds)  # ❌ incompatible type
    return cls


# type[T] w/o partialmethod
def pprint4(cls: type[T], /) -> type[T]:
    reveal_type(cls.__repr__)  # "(self: T@pprint4) -> str"
    cls.__repr__ = repr_fn  # ✅
    return cls
