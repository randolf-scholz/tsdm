r"""Callback protocols for TSDM."""

__all__ = [
    "BoundaryInformation",
    "OutlierBound",
    "OutlierInclusive",
    "SelectOutliersProto",
    "RemoveOutliersProto",
    "StripWhitespaceProto",
    # generic callback-protocols
    "NullMap",
    "SelfMap",
    "Lazy",
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

from collections.abc import Callable as Fn, Mapping
from typing import Any, Optional, Protocol, TypedDict

from numpy.typing import ArrayLike

from tsdm.types.aliases import Axis


class Lazy[T](Protocol):  # +T
    r"""Protocol for callables that take no arguments and return a value of type `T`."""

    def __call__(self, /) -> T: ...


class NullMap[T](Protocol):  # -T
    r"""Protocol for callables that take one argument and return `None`."""

    def __call__(self, x: T, /) -> None: ...


class SelfMap[T](Protocol):  # T
    r"""A generic protocol for endofunctions."""

    def __call__(self, x: T, /) -> T: ...


class CastProto[T](Protocol):  # T
    r"""Callback protocol for the `cast` function."""

    def __call__(self, x: T, /, dtype: Any) -> T: ...


class ClipProto[ArrayT, ScalarT = float](Protocol):  # T
    r"""Callback protocol for the `clip` function."""

    def __call__(
        self,
        x: ArrayT,
        lower: ArrayT | ScalarT | None,
        upper: ArrayT | ScalarT | None,
        /,
    ) -> ArrayT: ...


class ContractionProto[T](Protocol):  # T
    r"""Callback protocol for contractions accepting an `axis` keyword argument."""

    def __call__(self, x: T, /, *, axis: Axis = None) -> T | Any: ...


class CopyLikeProto[T](Protocol):  # T
    r"""Callback protocol for the `tensor_like` function."""

    def __call__(self, x: ArrayLike, ref: T, /) -> T: ...


class ToTensorProto[T](Protocol):  # +T
    r"""Callback protocol for the `to_tensor` function."""

    def __call__(self, x: ArrayLike, /) -> T: ...


class WhereProto[ArrayT, ScalarT = Any](Protocol):  # T
    r"""Callback protocol for the `where` function."""

    def __call__(
        self, cond: Any, x: ArrayT | ScalarT, y: ArrayT | ScalarT, /
    ) -> ArrayT: ...


class ApplyAlongAxes[T](Protocol):  # T
    r"""Callback protocol for the `apply_along_axes` function."""

    def __call__(self, op: Fn[..., T], /, *tensors: T, axis: Axis) -> T: ...


class ArraySplitProto[T](Protocol):  # T
    r"""Callback protocol for the `split_tensor` function."""

    def __call__(self, x: T, indices: int | list[int], /, *, axis: int) -> list[T]: ...


class ConcatenateProto[T](Protocol):  # T
    r"""Callback protocol for the `concatenate` function."""

    def __call__(self, x: list[T], /, *, axis: int) -> T: ...


class ScalarProto[T](Protocol):  # +T
    r"""Callback protocol for the `make_scalar` function."""

    def __call__(self, value: Any, /, dtype: Any) -> T: ...


class FullLikeProto[T](Protocol):  # T
    r"""Callback protocol for the `full_like` function."""

    def __call__(self, x: T, /, *, fill_value: Any) -> T: ...


class BoundaryInformation(TypedDict):
    r"""Bounds for one variable; `None` leaves a side unbounded."""

    lower_bound: float | None
    upper_bound: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None


type OutlierBound = float | Mapping[Any, float | None]
type OutlierInclusive = bool | Mapping[Any, bool | None]


class SelectOutliersProto[T, MaskT = T](Protocol):
    r"""Detect outliers using a limits table/mapping or four keyword bounds."""

    def __call__(
        self,
        obj: T,
        limits: Any = ...,
        /,
        *,
        lower_bound: Optional[OutlierBound] = ...,
        upper_bound: Optional[OutlierBound] = ...,
        lower_inclusive: Optional[OutlierInclusive] = ...,
        upper_inclusive: Optional[OutlierInclusive] = ...,
    ) -> MaskT: ...


class RemoveOutliersProto[T](Protocol):
    r"""Replace outliers with nulls, optionally dropping outlying rows."""

    def __call__(
        self,
        obj: T,
        limits: Any = ...,
        /,
        *,
        lower_bound: Optional[OutlierBound] = ...,
        upper_bound: Optional[OutlierBound] = ...,
        lower_inclusive: Optional[OutlierInclusive] = ...,
        upper_inclusive: Optional[OutlierInclusive] = ...,
        drop: bool = ...,
        inplace: bool = ...,
    ) -> T: ...


class StripWhitespaceProto[T](Protocol):
    r"""Trim string values in an object, optionally selecting table columns."""

    def __call__(self, obj: T, /, *cols: str) -> T: ...
