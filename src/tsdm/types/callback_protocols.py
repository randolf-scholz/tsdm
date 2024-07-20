r"""Callabck protocols for TSDM."""

__all__ = [
    # generic callback-protocols
    "IntMap",
    "NullMap",
    "SelfMap",
    "WrappedValue",
    # callback-protocols
    "ApplyAlongAxes",
    "ArraySplitProto",
    "CastProto",
    "ClipProto",
    "ConcatenateProto",
    "ContractionProto",
    "FullLikeProto",
    "IsScalarProto",
    "ScalarProto",
    "CopyLikeProto",
    "ToTensorProto",
    "WhereProto",
]

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, SupportsIndex, SupportsInt

from numpy.typing import ArrayLike

from tsdm.types.aliases import Axis, Scalar


# region generic callback-protocols ----------------------------------------------------
class NullMap[T](Protocol):  # -T
    r"""A generic protocol for functions without args that always returns None."""

    @abstractmethod
    def __call__(self, x: T, /) -> None:
        r"""Returns `None`."""
        ...


class SelfMap[T](Protocol):  # T
    r"""A generic protocol for endofunctions."""

    @abstractmethod
    def __call__(self, x: T, /) -> T:
        r"""Returns the result of the endofunction."""
        ...


class IntMap[T](Protocol):  # +T
    r"""A generic protocol for indexed values."""

    @abstractmethod
    def __call__(self, index: SupportsInt | SupportsIndex, /) -> T:
        r"""Returns the value at the given integer."""
        ...


class WrappedValue[T](Protocol):  # +T
    r"""A generic protocol for wrapped values."""

    @abstractmethod
    def __call__(self) -> T:
        r"""Returns the wrapped value."""
        ...


# endregion generic callback protocols -------------------------------------------------


# region Callback-Protocols ------------------------------------------------------------
class CastProto[T](Protocol):  # T
    r"""Bound-Protocol for `cast`-function."""

    def __call__(self, x: T, /, dtype: Any) -> T: ...


class ClipProto[T](Protocol):  # T
    r"""Bound-Protocol for `clip`-function."""

    def __call__(self, x: T, lower: T | None, upper: T | None, /) -> T: ...


class ContractionProto[T](Protocol):  # T
    r"""Bound Protocol for contractions (support `axes` keyword argument)."""

    def __call__(self, x: T, /, *, axis: Axis = None) -> T: ...


class IsScalarProto[T](Protocol):  # -T
    r"""Bound-Protocol for `is_scalar`-function."""

    def __call__(self, x: T, /) -> bool: ...


class CopyLikeProto[T](Protocol):  # T
    r"""Bound-Protocol for `tensor_like`-function."""

    def __call__(self, x: ArrayLike, ref: T, /) -> T: ...


class ToTensorProto[T](Protocol):  # +T
    r"""Callback-Protocol for `to_tensor`-function."""

    def __call__(self, x: ArrayLike, /) -> T: ...


class WhereProto[T](Protocol):  # T
    r"""Bound-Protocol for `where`-function."""

    def __call__(self, cond: T, x: T, y: Scalar | T, /) -> T: ...


class ApplyAlongAxes[T](Protocol):  # T
    r"""Bound-Protocol for `apply_along_axes`-function."""

    def __call__(self, op: Callable[..., T], /, *tensors: T, axis: Axis) -> T: ...


class ArraySplitProto[T](Protocol):  # T
    r"""Bound-Protocol for `split_tensor`-function."""

    def __call__(self, x: T, indices: int | list[int], /, *, axis: int) -> list[T]: ...


class ConcatenateProto[T](Protocol):  # T
    r"""Bound-Protocol for `concatenate`-function."""

    def __call__(self, x: list[T], /, *, axis: int) -> T: ...


class ScalarProto[T](Protocol):  # +T
    r"""Bound-Protocol for `make_scalar`-function."""

    def __call__(self, value: Any, /, dtype: Any) -> T: ...


class FullLikeProto[T](Protocol):  # T
    r"""Bound-Protocol for `full_like`-function."""

    def __call__(self, x: T, /, *, fill_value: Any) -> T: ...


# endregion Callback-Protocols ---------------------------------------------------------
