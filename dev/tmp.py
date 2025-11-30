from typing import Callable, Protocol


class Decorator[**P](Protocol):
    def __call__(self, **kwargs: P.kwargs) -> None: ...


def make[**P](fn: Callable[P, None]) -> Decorator[P]: ...


def demo(a: int = 1, /, *, b: int = 2) -> None: ...


fn = make(demo)
fn(b=1)
fn(c=1)
fn(1)
