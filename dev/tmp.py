#!/usr/bin/env python

from collections.abc import Callable
from typing import Any, ParamSpec, Protocol, TypeVar, reveal_type, runtime_checkable

R = TypeVar("R", covariant=True)
P = ParamSpec("P")


@runtime_checkable
class Func(Protocol):
    """Protocol for functions, alternative to `Callable`."""

    __call__: Callable[..., Any]

    # def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    # def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


# test cases
class Baz:
    def __call__(self) -> None:
        return


def foo() -> None: ...


bar = lambda: None
baz = Baz()


# try to match against `Callable`
match foo:
    case Callable() as func:  # ❌ Expected type got typing._SpecialForm
        reveal_type(func)  # ❌   Statement is unreachable

match bar:
    case Callable() as func:  # ❌ Expected type got typing._SpecialForm
        reveal_type(func)  # ❌   Statement is unreachable

match baz:
    case Callable() as func:  # ❌ Expected type got typing._SpecialForm
        reveal_type(func)  # ❌   Statement is unreachable

# try to match against `Func`
match foo:
    case Func() as func:  # ✅
        reveal_type(func)  # ❌   Statement is unreachable


match bar:
    case Func() as func:  # ✅
        reveal_type(func)  # ❌   Statement is unreachable


match baz:
    case Func() as func:  # ✅
        reveal_type(func)  # ✅


gg: Func = lambda: None  # ✅


class B: ...


assert issubclass(type(lambda: None), Func)  # ✅
# assert issubclass(B, Func)
