# https://github.com/python/mypy/issues/20720
# https://github.com/python/typing/issues/2021

from typing import Protocol, overload


def test_mutually_exclusive_signatures() -> None:
    class X: ...

    class Y: ...

    class Proto(Protocol):
        @overload
        def op(self, *, x: X) -> None: ...
        @overload
        def op(self, *, y: Y) -> None: ...

    class ImplA:
        @overload
        def op(self, *, x: X) -> None: ...
        @overload
        def op(self, *, y: Y) -> None: ...

    class ImplB:
        @overload
        def op(self, *, y: Y) -> None: ...
        @overload
        def op(self, *, x: X) -> None: ...

    def _check(a: ImplA, b: ImplB) -> None:
        _0: Proto = a  # OK
        _1: Proto = b  # rejected by mypy


def test_same_signature_and_return() -> None:
    class X: ...

    class Y: ...

    class Proto(Protocol):
        @overload
        def op(self, x: X, /) -> None: ...
        @overload
        def op(self, y: Y, /) -> None: ...

    class ImplA:
        @overload
        def op(self, x: X, /) -> None: ...
        @overload
        def op(self, y: Y, /) -> None: ...

    class ImplB:
        @overload
        def op(self, y: Y, /) -> None: ...
        @overload
        def op(self, x: X, /) -> None: ...

    class ImplC:
        def op(self, z: X | Y, /) -> None: ...

    def _check(a: ImplA, b: ImplB, c: ImplC) -> None:
        _0: Proto = a  # OK
        _1: Proto = b  # rejected
        _2: Proto = c  # OK


def test_mutually_exclusive_types() -> None:
    class Proto(Protocol):
        @overload
        def op(self, x: int, /) -> int: ...
        @overload
        def op(self, y: str, /) -> str: ...

    class ImplA:
        @overload
        def op(self, x: int, /) -> int: ...
        @overload
        def op(self, y: str, /) -> str: ...

    class ImplB:
        @overload
        def op(self, y: str, /) -> str: ...
        @overload
        def op(self, x: int, /) -> int: ...

    class ImplC:
        def op[T: (int, str)](self, z: T, /) -> T: ...

    def _check(a: ImplA, b: ImplB, c: ImplC) -> None:
        _0: Proto = a  # OK
        _1: Proto = b  # rejected
        _2: Proto = c  # OK


def test_mutually_exclusive_custom_types() -> None:
    from typing_extensions import disjoint_base

    @disjoint_base
    class X: ...

    @disjoint_base
    class Y: ...

    class Proto(Protocol):
        @overload
        def op(self, x: X, /) -> X: ...
        @overload
        def op(self, y: Y, /) -> Y: ...

    class ImplA:
        @overload
        def op(self, x: X, /) -> X: ...
        @overload
        def op(self, y: Y, /) -> Y: ...

    class ImplB:
        @overload
        def op(self, y: Y, /) -> Y: ...
        @overload
        def op(self, x: X, /) -> X: ...

    def _check(a: ImplA, b: ImplB) -> None:
        _0: Proto = a  # OK
        _1: Proto = b  # rejected
