r"""Callabck protocols for TSDM."""

__all__ = [
    # generic callback-protocols
    "IntMap",
    "NullMap",
    "SelfMap",
    "WrappedValue",
    # callback-protocols
    "ApplyAlongAxes",
    "CastProto",
    "ClipProto",
    "ConcatenateProto",
    "ContractionProto",
    "IsScalarProto",
    "ArraySplitProto",
    "TensorLikeProto",
    "ToTensorProto",
    "WhereProto",
    "MakeScalarProto",
]

from abc import abstractmethod
from collections.abc import Callable

from numpy.typing import ArrayLike
from typing_extensions import Any, Protocol, SupportsIndex, SupportsInt

from tsdm.types.aliases import Axis, Scalar
from tsdm.types.variables import T, T_co, T_contra


# region generic callback-protocols ----------------------------------------------------
class NullMap(Protocol[T_contra]):
    r"""A generic protocol for functions without args that always returns None."""

    @abstractmethod
    def __call__(self, x: T_contra, /) -> None:
        r"""Returns `None`."""
        ...


class SelfMap(Protocol[T]):
    r"""A generic protocol for endofunctions."""

    @abstractmethod
    def __call__(self, x: T, /) -> T:
        r"""Returns the result of the endofunction."""
        ...


class IntMap(Protocol[T_co]):
    r"""A generic protocol for indexed values."""

    @abstractmethod
    def __call__(self, index: SupportsInt | SupportsIndex, /) -> T_co:
        r"""Returns the value at the given integer."""
        ...


class WrappedValue(Protocol[T_co]):
    r"""A generic protocol for wrapped values."""

    @abstractmethod
    def __call__(self) -> T_co:
        r"""Returns the wrapped value."""
        ...


# endregion generic callback protocols -------------------------------------------------


# region Callback-Protocols ------------------------------------------------------------
class CastProto(Protocol[T]):
    r"""Bound-Protocol for `cast`-function."""

    def __call__(self, x: T, /, dtype: Any) -> T: ...


class ClipProto(Protocol[T]):
    r"""Bound-Protocol for `clip`-function."""

    def __call__(self, x: T, lower: T | None, upper: T | None, /) -> T: ...


class ContractionProto(Protocol[T]):
    r"""Bound Protocol for contractions (support `axes` keyword argument)."""

    def __call__(self, x: T, /, *, axis: Axis = None) -> T: ...


class IsScalarProto(Protocol[T_contra]):
    r"""Bound-Protocol for `is_scalar`-function."""

    def __call__(self, x: T_contra, /) -> bool: ...


class TensorLikeProto(Protocol[T]):
    r"""Bound-Protocol for `tensor_like`-function."""

    def __call__(self, x: ArrayLike, ref: T, /) -> T: ...


class ToTensorProto(Protocol[T_co]):
    r"""Callback-Protocol for `to_tensor`-function."""

    def __call__(self, x: ArrayLike, /) -> T_co: ...


class WhereProto(Protocol[T]):
    r"""Bound-Protocol for `where`-function."""

    def __call__(self, cond: T, x: T, y: Scalar | T, /) -> T: ...


class ApplyAlongAxes(Protocol[T]):
    r"""Bound-Protocol for `apply_along_axes`-function."""

    def __call__(self, op: Callable[..., T], /, *tensors: T, axis: Axis) -> T: ...


class ArraySplitProto(Protocol[T]):
    r"""Bound-Protocol for `split_tensor`-function."""

    def __call__(self, x: T, indices: int | list[int], /, *, axis: int) -> list[T]: ...


class ConcatenateProto(Protocol[T]):
    r"""Bound-Protocol for `concatenate`-function."""

    def __call__(self, x: list[T], /, *, axis: int) -> T: ...


class MakeScalarProto(Protocol[T_co]):
    r"""Bound-Protocol for `make_scalar`-function."""

    def __call__(self, value: Any, /, dtype: Any) -> T_co: ...


# endregion Callback-Protocols ---------------------------------------------------------
