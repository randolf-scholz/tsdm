As a workaround, I tried using a `Protocol` that emulates `Callable`, but it still results in a bunch of `[unreachable]` errors

```python
from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeVar, reveal_type, runtime_checkable

R = TypeVar("R", covariant=True)
P = ParamSpec("P")


@runtime_checkable
class Func(Protocol[P, R]):
    """Protocol for functions, alternative to `Callable`."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute the callable."""
        ...


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
```

https://mypy-play.net/?mypy=latest&python=3.12&flags=strict%2Cwarn-unreachable&gist=f90a2f8b6bc47bc57c01be7538aa242b
