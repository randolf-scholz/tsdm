"""Protocols for backend-functions."""

__all__ = [
    # Type aliases
    "Scalar",
    # Protocols
    "Clip",
    "ClipProto",
    "Contraction",
    "ContractionProto",
    "IsScalar",
    "IsScalarProto",
    "TensorLike",
    "TensorLikeProto",
    "ToTensor",
    "ToTensorProto",
    "Where",
    "WhereProto",
]

from typing import Protocol, TypeAlias, overload

from numpy.typing import ArrayLike

from tsdm.types.aliases import Axes
from tsdm.types.variables import any_co as T_co, any_contra as T_contra, any_var as T

Scalar: TypeAlias = bool | int | float | complex
"""A type alias for scalar values."""


class Clip(Protocol):
    """Callback-Protocol for `clip`-function."""

    def __call__(self, x: T, lower: T | None, upper: T | None, /) -> T:
        ...


class ClipProto(Protocol[T]):
    """Bound-Protocol for `clip`-function."""

    def __call__(self, x: T, lower: T | None, upper: T | None, /) -> T:
        ...


class Contraction(Protocol):
    """Callback-Protocol for contractions (support `axes` keyword argument)."""

    def __call__(self, __x: T, *, axis: Axes = None) -> T:
        ...


class ContractionProto(Protocol[T]):
    """Bound Protocol for contractions (support `axes` keyword argument)."""

    def __call__(self, __x: T, *, axis: Axes = None) -> T:
        ...


class IsScalar(Protocol):
    """Callback-Protocol for `is_scalar`-function."""

    def __call__(self, x: T_contra, /) -> bool:
        ...


class IsScalarProto(Protocol[T_contra]):
    """Bound-Protocol for `is_scalar`-function."""

    def __call__(self, x: T_contra, /) -> bool:
        ...


class TensorLike(Protocol):
    """Callback-Protocol for `tensor_like`-function."""

    def __call__(self, x: ArrayLike, ref: T, /) -> T:
        ...


class TensorLikeProto(Protocol[T]):
    """Bound-Protocol for `tensor_like`-function."""

    def __call__(self, x: ArrayLike, ref: T, /) -> T:
        ...


class ToTensorProto(Protocol[T_co]):
    """Callback-Protocol for `to_tensor`-function."""

    def __call__(self, x: ArrayLike, /) -> T_co:
        ...

    # @overload
    # def __call__(self, x: T, /) -> T:
    #     ...


class ToTensor(Protocol[T_co]):
    """Bound-Protocol for `to_tensor`-function."""

    @overload
    def __call__(self, x: ArrayLike, /) -> T_co:
        ...

    @overload
    def __call__(self, x: T, /) -> T:
        ...


class Where(Protocol):
    """Callback-Protocol for `where`-function."""

    def __call__(self, cond: T, x: T, y: Scalar | T, /) -> T:
        ...


class WhereProto(Protocol[T]):
    """Bound-Protocol for `where`-function."""

    def __call__(self, cond: T, x: T, y: Scalar | T, /) -> T:
        ...
