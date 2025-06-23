r"""Callabck protocols for TSDM."""

__all__ = [
    # generic callback-protocols
    "IntMap",
    "NullMap",
    "SelfMap",
    "Lazy",
    "IdentityMap",
    "IdentityMapOnFn",
    "IdentityMapOnCls",
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

from collections.abc import Callable as Fn
from typing import Any, Protocol, SupportsIndex

from numpy.typing import ArrayLike

from tsdm.types.aliases import Axis, BuiltinScalar


# region generic callback-protocols ----------------------------------------------------
class IdentityMap(Protocol):
    r"""Protocol for Identity functions."""

    def __call__[T](self, obj: T, /) -> T: ...


# FIXME: use IdentityMap[type] once bounding T-vars by other T-vars is supported
class IdentityMapOnCls(Protocol):
    r"""Protocol for class decorators that return the same class."""

    def __call__[Cls: type](self, cls: Cls, /) -> Cls: ...


# FIXME: use IdentityMap[Fn] once bounding T-vars by other T-vars is supported
class IdentityMapOnFn(Protocol):
    r"""Protocol for function decorators that return the same function."""

    def __call__[F: Fn](self, fn: F, /) -> F: ...


class NullMap[T](Protocol):  # -T
    r"""A generic protocol for functions without args that always returns None."""

    def __call__(self, x: T, /) -> None: ...


class SelfMap[T](Protocol):  # T
    r"""A generic protocol for endofunctions."""

    def __call__(self, x: T, /) -> T: ...


class IntMap[T](Protocol):  # +T
    r"""A generic protocol for indexed values."""

    def __call__(self, index: SupportsIndex, /) -> T: ...


class Lazy[T](Protocol):  # +T
    r"""A generic protocol for wrapped values."""

    def __call__(self, /) -> T: ...


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

    def __call__(self, cond: T, x: T, y: BuiltinScalar | T, /) -> T: ...


class ApplyAlongAxes[T](Protocol):  # T
    r"""Bound-Protocol for `apply_along_axes`-function."""

    def __call__(self, op: Fn[..., T], /, *tensors: T, axis: Axis) -> T: ...


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
