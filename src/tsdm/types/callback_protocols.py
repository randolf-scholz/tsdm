"""Callabck protocols for TSDM."""

__all__ = [
    # generic
    "Func",
    # generic callback-protocols
    "IntMap",
    "NullMap",
    "SelfMap",
    "WrappedValue",
    # callback-protocols
    "ApplyAlongAxes",
    "ClipProto",
    "ContractionProto",
    "IsScalarProto",
    "TensorLikeProto",
    "ToTensorProto",
    "WhereProto",
]

from collections.abc import Callable

from numpy.typing import ArrayLike
from typing_extensions import Protocol, SupportsIndex, SupportsInt, runtime_checkable

from tsdm.types.aliases import Axes, Scalar
from tsdm.types.protocols import P
from tsdm.types.variables import (
    any_co as T_co,
    any_contra as T_contra,
    any_var as T,
    return_var_co as R,
)


@runtime_checkable
class Func(Protocol[P, R]):
    """Protocol for functions, alternative to `Callable`."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


# region generic callback-protocols ----------------------------------------------------
class NullMap(Protocol[T_contra]):
    r"""A generic protocol for functions without args that always returns None."""

    def __call__(self, x: T_contra, /) -> None:
        r"""Returns `None`."""
        ...


class SelfMap(Protocol[T]):
    r"""A generic protocol for endofunctions."""

    def __call__(self, x: T, /) -> T:
        r"""Returns the result of the endofunction."""
        ...


class IntMap(Protocol[T_co]):
    r"""A generic protocol for indexed values."""

    def __call__(self, index: SupportsInt | SupportsIndex, /) -> T_co:
        r"""Returns the value at the given integer."""
        ...


class WrappedValue(Protocol[T_co]):
    r"""A generic protocol for wrapped values."""

    def __call__(self) -> T_co:
        r"""Returns the wrapped value."""
        ...


# endregion generic callback protocols -------------------------------------------------


# region Callback-Protocols ------------------------------------------------------------
class ClipProto(Protocol[T]):
    """Bound-Protocol for `clip`-function."""

    def __call__(self, x: T, lower: T | None, upper: T | None, /) -> T: ...


class ContractionProto(Protocol[T]):
    """Bound Protocol for contractions (support `axes` keyword argument)."""

    def __call__(self, x: T, /, *, axis: Axes = None) -> T: ...


class IsScalarProto(Protocol[T_contra]):
    """Bound-Protocol for `is_scalar`-function."""

    def __call__(self, x: T_contra, /) -> bool: ...


class TensorLikeProto(Protocol[T]):
    """Bound-Protocol for `tensor_like`-function."""

    def __call__(self, x: ArrayLike, ref: T, /) -> T: ...


class ToTensorProto(Protocol[T_co]):
    """Callback-Protocol for `to_tensor`-function."""

    def __call__(self, x: ArrayLike, /) -> T_co: ...


class WhereProto(Protocol[T]):
    """Bound-Protocol for `where`-function."""

    def __call__(self, cond: T, x: T, y: Scalar | T, /) -> T: ...


class ApplyAlongAxes(Protocol[T]):
    """Bound-Protocol for `apply_along_axes`-function."""

    def __call__(self, op: Callable[..., T], /, *tensors: T, axis: Axes) -> T: ...


# endregion Callback-Protocols ---------------------------------------------------------
