from typing import Callable, Generic, Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class NullMap(Protocol[T_contra]):
    def __call__(self, x: T_contra, /) -> None: ...


class Foo(Generic[T_co]):
    def init(self, x: T_co, /) -> None: ...  # mypy: ❌, pyright: ❌


class Bar(Generic[T_co]):
    @property
    def initializer(self) -> Callable[[T_co], None]:  # mypy: ❌, pyright: ✅
        return lambda x: None


class Baz(Generic[T_co]):
    @property
    def initializer(self) -> NullMap[T_co]:  # mypy: ✅, pyright: ✅
        return lambda x: None


# Example LSP violation

from typing import Callable, Generic, TypeVar


class A:
    def reset(self) -> None: ...


T_co = TypeVar("T_co")
A_co = TypeVar("A_co", bound=A)


class Example(Generic[T_co]):
    @property
    def initializer(self) -> Callable[[T_co], None]:
        return lambda _: None


class SubExample(Example[A_co]):
    @property
    def initializer(self) -> Callable[[A_co], None]:
        return lambda x: x.reset()


def foo(example: Example[object]) -> None:
    x = object()
    example.initializer(x)


foo(SubExample())  # TypeError at runtime
