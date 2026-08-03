from typing import Protocol, Self, overload


class X: ...


class Y: ...


def test_can_self() -> None:
    class CanSelfOP(Protocol):
        def __op__(self, other: Self, /) -> Self: ...

    class A:  # (A) -> A
        def __op__(self, other: A, /) -> A: ...

    class A_Self:  # (Self) -> Self
        def __op__(self, other: Self, /) -> Self: ...

    def test_assign(a_impl: A, a_self: A_Self) -> None:
        _0: CanSelfOP = a_impl
        _1: CanSelfOP = a_self


def test_can_self_overloaded_implementation() -> None:
    class CanSelfOP(Protocol):
        def __op__(self, other: Self, /) -> Self: ...

    class A:  # (X -> A), (A -> A)
        @overload
        def __op__(self, other: X, /) -> A: ...
        @overload
        def __op__(self, other: A, /) -> A: ...

    class A_Self:  # (X -> Self), (Self -> Self)
        @overload
        def __op__(self, other: X, /) -> Self: ...
        @overload
        def __op__(self, other: Self, /) -> Self: ...

    class B:  # (B -> B), (X -> B)
        @overload
        def __op__(self, other: B, /) -> B: ...
        @overload
        def __op__(self, other: X, /) -> B: ...

    class B_Self:  # (Self -> Self), (X -> Self)
        @overload
        def __op__(self, other: Self, /) -> Self: ...
        @overload
        def __op__(self, other: X, /) -> Self: ...

    def test_assign(a_impl: A, a_self: A_Self, b_impl: B, b_self: B_Self) -> None:
        _0: CanSelfOP = a_impl
        _1: CanSelfOP = a_self
        _2: CanSelfOP = b_impl
        _3: CanSelfOP = b_self


def test_can_self_overloaded_protocol() -> None:
    class CanSelfOP(Protocol):  # (X -> Self), (Self -> Self)
        @overload
        def __op__(self, other: X, /) -> Self: ...
        @overload
        def __op__(self, other: Self, /) -> Self: ...

    class CanSelfOP_Alt(Protocol):  # (Self -> Self), (X -> Self)
        @overload
        def __op__(self, other: Self, /) -> Self: ...
        @overload
        def __op__(self, other: X, /) -> Self: ...

    class CanSelfOP_Union(Protocol):  # (X | Self) -> Self
        def __op__(self, other: X | Self, /) -> Self: ...

    class A:  #  (A | X) -> A
        def __op__(self, other: A | X, /) -> A: ...

    class A_Self:  # (Self | X) -> Self
        def __op__(self, other: Self | X, /) -> Self: ...

    class B:  # (B -> B), (X -> B)
        @overload
        def __op__(self, other: B, /) -> B: ...
        @overload
        def __op__(self, other: X, /) -> B: ...

    class B_Self:  # (Self -> Self), (X -> Self)
        @overload
        def __op__(self, other: Self, /) -> Self: ...
        @overload
        def __op__(self, other: X, /) -> Self: ...

    class C:  # (X -> C), (C -> C)
        @overload
        def __op__(self, other: X, /) -> C: ...
        @overload
        def __op__(self, other: C, /) -> C: ...

    class C_Self:  # (X -> Self), (Self -> Self)
        @overload
        def __op__(self, other: X, /) -> Self: ...
        @overload
        def __op__(self, other: Self, /) -> Self: ...

    def test_assign(
        a_impl: A,
        a_self: A_Self,
        b_impl: B,
        b_self: B_Self,
        c_impl: C,
        c_self: C_Self,
    ) -> None:
        _x0: CanSelfOP = a_impl
        _x1: CanSelfOP = a_self
        _x2: CanSelfOP = b_impl  # ❌️: mypy
        _x3: CanSelfOP = b_self  # ❌️: mypy
        _x4: CanSelfOP = c_impl
        _x5: CanSelfOP = c_self

        _y0: CanSelfOP_Alt = a_impl
        _y1: CanSelfOP_Alt = a_self
        _y2: CanSelfOP_Alt = b_impl
        _y3: CanSelfOP_Alt = b_self
        _y4: CanSelfOP_Alt = c_impl  # ❌️: mypy
        _y5: CanSelfOP_Alt = c_self  # ❌️: mypy

        _z0: CanSelfOP_Union = a_impl
        _z1: CanSelfOP_Union = a_self
        _z2: CanSelfOP_Union = b_impl  # type: ignore[assignment] # ❌️: mypy, pyright, pyrefly, ty
        _z3: CanSelfOP_Union = b_self  # type: ignore[assignment] # ❌️: mypy, pyright, pyrefly, ty
        _z4: CanSelfOP_Union = c_impl  # type: ignore[assignment] # ❌️: mypy, pyright, pyrefly, ty
        _z5: CanSelfOP_Union = c_self  # type: ignore[assignment] # ❌️: mypy, pyright, pyrefly, ty
