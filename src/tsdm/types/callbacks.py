r"""Callabck protocols for TSDM."""

__all__ = [
    # generic callback-protocols
    "NullMap",
    "SelfMap",
    "Lazy",
    "Polymorphism",
    # callback-protocols
    "ApplyAlongAxes",
    "ArraySplitProto",
    "CastProto",
    "ClipProto",
    "ConcatenateProto",
    "ContractionProto",
    "FullLikeProto",
    "ScalarProto",
    "CopyLikeProto",
    "ToTensorProto",
    "WhereProto",
]

from collections.abc import Callable as Fn
from typing import Any, Protocol

from numpy.typing import ArrayLike

from .aliases import Axis


# region generic callback-protocols ----------------------------------------------------
class Lazy[T](Protocol):  # +T
    r"""A generic protocol for wrapped values."""

    def __call__(self, /) -> T: ...


class Polymorphism[**P = []](Protocol):
    r"""Protocol for Identity functions."""

    def __call__[T: Any](self, obj: T, /, *arg: P.args, **kwargs: P.kwargs) -> T: ...


class NullMap[T](Protocol):  # -T
    r"""A generic protocol for functions without args that always returns None."""

    def __call__(self, x: T, /) -> None: ...


class SelfMap[T](Protocol):  # T
    r"""A generic protocol for endofunctions."""

    def __call__(self, x: T, /) -> T: ...


# endregion generic callback protocols -------------------------------------------------


# region Callback-Protocols ------------------------------------------------------------
class CastProto[T](Protocol):  # T
    r"""Bound-Protocol for `cast`-function."""

    def __call__(self, x: T, /, dtype: Any) -> T: ...


class ClipProto[ArrayT, ScalarT = float](Protocol):  # T
    r"""Bound-Protocol for `clip`-function."""

    def __call__(
        self,
        x: ArrayT,
        lower: ArrayT | ScalarT | None,
        upper: ArrayT | ScalarT | None,
        /,
    ) -> ArrayT: ...


class ContractionProto[T](Protocol):  # T
    r"""Bound Protocol for contractions (support `axis` keyword argument)."""

    def __call__(self, x: T, /, *, axis: Axis = None) -> T | Any: ...


class CopyLikeProto[T](Protocol):  # T
    r"""Bound-Protocol for `tensor_like`-function."""

    def __call__(self, x: ArrayLike, ref: T, /) -> T: ...


class ToTensorProto[T](Protocol):  # +T
    r"""Callback-Protocol for `to_tensor`-function."""

    def __call__(self, x: ArrayLike, /) -> T: ...


class WhereProto[ArrayT, ScalarT = Any](Protocol):  # T
    r"""Bound-Protocol for `where`-function."""

    def __call__(
        self, cond: Any, x: ArrayT | ScalarT, y: ArrayT | ScalarT, /
    ) -> ArrayT: ...


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
