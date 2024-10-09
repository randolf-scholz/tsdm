r"""Encoders for ensuring bounds on the input data."""

__all__ = ["BoundaryEncoder"]

from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from typing import Any, Generic, Literal, Optional, Self

import pandas as pd
from typing_extensions import TypeVar

from tsdm.backend import Backend, get_backend
from tsdm.encoders import BaseEncoder
from tsdm.types.arrays import NumericalSeries
from tsdm.types.scalars import OrderedScalar
from tsdm.utils.decorators import pprint_repr

# FIXME: python==3.13: use PEP695 with default values.
S = TypeVar("S", bound=OrderedScalar)
Arr = TypeVar("Arr", bound=NumericalSeries, default=NumericalSeries[S])


@pprint_repr
@dataclass
class BoundaryEncoder(BaseEncoder[Arr, Arr], Generic[S, Arr]):
    r"""Clip or mask values outside a given range.

    Args:
        lower_bound: the lower boundary. If not provided, it is determined by the mask/data.
        upper_bound: the upper boundary. If not provided, it is determined by the mask/data.
        lower_included: whether the lower boundary is included in the range.
        upper_included: whether the upper boundary is included in the range.
        mode: one of ``'mask'`` or ``'clip'``, or a tuple of two of them for lower and upper.
            - If `mode='mask'`, then values outside the boundary will be replaced by `NA`.
            - If `mode='clip'`, then values outside the boundary will be clipped to it.

    Note:
        requires_fit: whether the data should determine the lower/upper bounds/values.
            - if lower/upper not provided, then they are determined by the data.
            - if lower_substitute/upper_substitute not provided, then they are determined by the data.

    Examples:
        - `BoundaryEncoder()` will mask values outside the range `[data_min, data_max]`
        - `BoundaryEncoder(mode='clip')` will clip values to the range `[data_min, data_max]`
        - `BoundaryEncoder(0, 1)` will mask values outside the range `[0,1]`
        - `BoundaryEncoder(0, 1, mode='clip')` will clip values to the range `[0,1]`
        - `BoundaryEncoder(0, 1, mode=('mask', 'clip'))` will mask values below 0 and clip values above 1 to 1.
        - `BoundaryEncoder(0, mode=('mask', 'clip'))` will mask values below 0 and clip values above 1 to `data_max`.
    """

    class CLIPPING(StrEnum):
        r"""Type Hint for clipping mode."""

        mask = "mask"
        clip = "clip"

    type ClippingMode = Literal["mask", "clip"]
    r"""Type Hint for clipping mode."""

    lower_bound: Optional[S] = NotImplemented
    upper_bound: Optional[S] = NotImplemented

    _: KW_ONLY

    lower_included: bool = True
    upper_included: bool = True
    lower_mode: CLIPPING = NotImplemented
    upper_mode: CLIPPING = NotImplemented

    # derived attributes
    backend: Backend = field(init=False)
    lower_value: S = field(init=False, default=NotImplemented)
    upper_value: S = field(init=False, default=NotImplemented)

    def __init__(
        self,
        lower_bound: Optional[S] = NotImplemented,
        upper_bound: Optional[S] = NotImplemented,
        *,
        lower_included: bool = True,
        upper_included: bool = True,
        mode: CLIPPING | str | tuple[CLIPPING | str, CLIPPING | str] = "mask",
    ) -> None:
        r"""Initialize the BoundaryEncoder."""
        self.lower_included = lower_included
        self.upper_included = upper_included

        try:  # set the lower_bound
            lower_is_nan = pd.isna(lower_bound)
        except (TypeError, ValueError):
            # Raises if upper_bound is an array
            self.lower_bound = lower_bound
        else:
            self.lower_bound = None if lower_is_nan else lower_bound

        try:  # set the upper_bound
            upper_is_nan = pd.isna(upper_bound)
        except (TypeError, ValueError):
            # Raises if upper_bound is an array
            self.upper_bound = upper_bound
        else:
            self.upper_bound = None if upper_is_nan else upper_bound

        match mode:
            case (self.CLIPPING() | str()) as value:
                self.lower_mode = self.CLIPPING(value)
                self.upper_mode = self.CLIPPING(value)
            case [
                (self.CLIPPING() | str()) as lower,
                (self.CLIPPING() | str()) as upper,
            ]:
                self.lower_mode = self.CLIPPING(lower)
                self.upper_mode = self.CLIPPING(upper)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        # validate internal consistency
        if (
            self.upper_bound is not None
            and self.upper_bound is not NotImplemented
            and self.lower_bound is not None
            and self.lower_bound is not NotImplemented
            and self.upper_bound <= self.lower_bound
        ):
            raise ValueError("lower_bound must be smaller than upper_bound.")

    @classmethod
    def from_interval(cls, interval: pd.Interval, **kwargs: Any) -> Self:
        r"""Create a BoundaryEncoder from a pandas Interval."""
        lower_bound = interval.left
        upper_bound = interval.right
        lower_included, upper_included = {
            "left": (True, False),
            "right": (False, True),
            "both": (True, True),
            "neither": (False, False),
        }[interval.closed]
        return cls(
            lower_bound,
            upper_bound,
            lower_included=lower_included,
            upper_included=upper_included,
            **kwargs,
        )

    def lower_satisfied(self, x: Arr) -> Arr:
        r"""Return a boolean mask for the lower boundary (true: value ok)."""
        if self.lower_bound is None:
            return self.backend.true_like(x)
        if self.lower_included or self.lower_included is None:
            return (x >= self.lower_bound) | self.backend.is_null(x)
        return (x > self.lower_bound) | self.backend.is_null(x)

    def upper_satisfied(self, x: Arr) -> Arr:
        r"""Return a boolean mask for the upper boundary (true: value ok)."""
        if self.upper_bound is None:
            return self.backend.true_like(x)
        if self.upper_included or self.upper_included is None:
            return (x <= self.upper_bound) | self.backend.is_null(x)
        return (x < self.upper_bound) | self.backend.is_null(x)

    def _fit_impl(self, data: Arr, /) -> None:
        # select the backend
        self.backend: Backend = get_backend(data)

        # fit the parameters
        if self.lower_bound is NotImplemented:
            self.lower_bound = self.backend.nanmin(data)
        if self.upper_bound is NotImplemented:
            self.upper_bound = self.backend.nanmax(data)

        # set lower_value
        match self.lower_bound, self.lower_mode:
            case None, _:
                self.lower_value = self.backend.to_tensor(float("-inf"))
            case _, self.CLIPPING.mask:
                self.lower_value = self.backend.to_tensor(float("nan"))
            case _, self.CLIPPING.clip:
                self.lower_value = self.lower_bound  # type: ignore[assignment]
            case _:
                raise NotImplementedError

        # set upper_value
        match self.upper_bound, self.upper_mode:
            case None, _:
                self.upper_value = self.backend.to_tensor(float("+inf"))
            case _, self.CLIPPING.mask:
                self.upper_value = self.backend.to_tensor(float("nan"))
            case _, self.CLIPPING.clip:
                self.upper_value = self.upper_bound  # type: ignore[assignment]
            case _:
                raise NotImplementedError

    def _encode_impl(self, data: Arr, /) -> Arr:
        # NOTE: frame.where(cond, other) replaces with other if condition is false!
        data = self.backend.where(self.lower_satisfied(data), data, self.lower_value)
        data = self.backend.where(self.upper_satisfied(data), data, self.upper_value)
        return data

    def _decode_impl(self, data: Arr, /) -> Arr:
        return data
