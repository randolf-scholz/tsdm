r"""Encoders for ensuring bounds on the input data."""

__all__ = ["BoundaryEncoder"]

from dataclasses import KW_ONLY, dataclass, field
from enum import StrEnum
from typing import Any, Literal, Optional, Self

import pandas as pd

from tsdm.backend import Backend, get_backend
from tsdm.backend.fallback import is_null_scalar
from tsdm.backend.types import NumericalSeries
from tsdm.constants import UNDEFINED
from tsdm.encoders.base import FittableEncoder
from tsdm.types.scalars import OrderedScalar
from tsdm.utils.decorators import pprint_repr


@pprint_repr
@dataclass
class BoundaryEncoder[
    S: OrderedScalar = float,
    Arr: NumericalSeries = NumericalSeries[S],
](FittableEncoder[Arr, Arr]):
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

    class MODES(StrEnum):
        r"""Type Hint for clipping mode."""

        mask = "mask"
        clip = "clip"

    type Mode = Literal["mask", "clip"]
    r"""Type Hint for clipping mode."""

    lower_bound: Optional[S] = UNDEFINED
    upper_bound: Optional[S] = UNDEFINED

    _: KW_ONLY

    lower_included: bool = True
    upper_included: bool = True
    lower_mode: MODES = UNDEFINED
    upper_mode: MODES = UNDEFINED

    # derived attributes
    backend: Backend = field(init=False, default=UNDEFINED)
    lower_value: S = field(init=False, default=UNDEFINED)
    upper_value: S = field(init=False, default=UNDEFINED)

    def __init__(
        self,
        lower_bound: Optional[S] = UNDEFINED,
        upper_bound: Optional[S] = UNDEFINED,
        *,
        lower_included: bool = True,
        upper_included: bool = True,
        mode: MODES | str | tuple[MODES | str, MODES | str] = "mask",
    ) -> None:
        r"""Initialize the BoundaryEncoder."""
        self.lower_included = lower_included
        self.upper_included = upper_included
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        match mode:
            case (self.MODES() | str()) as value:
                self.lower_mode = self.MODES(value)
                self.upper_mode = self.MODES(value)
            case [
                (self.MODES() | str()) as lower,
                (self.MODES() | str()) as upper,
            ]:
                self.lower_mode = self.MODES(lower)
                self.upper_mode = self.MODES(upper)
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    def _validate(self) -> None:
        r"""Validate the encoder configuration."""
        if (
            self.upper_bound is not None
            and self.upper_bound is not UNDEFINED
            and self.lower_bound is not None
            and self.lower_bound is not UNDEFINED
            and self.upper_bound <= self.lower_bound
        ):
            raise ValueError("lower_bound must be smaller than upper_bound.")

        if self.lower_included and self.lower_mode == self.MODES.clip:
            raise ValueError(
                "Incompatible combination: lower_included=True and lower_mode='clip'."
            )
        if self.upper_included and self.upper_mode == self.MODES.clip:
            raise ValueError(
                "Incompatible combination: upper_included=True and upper_mode='clip'."
            )

    @classmethod
    def from_interval(cls, interval: pd.Interval, **kwargs: Any) -> Self:
        r"""Create a BoundaryEncoder from a pandas Interval."""
        lower_bound = interval.left
        upper_bound = interval.right
        lower_included, upper_included = {
            "left":    (True, False),
            "right":   (False, True),
            "both":    (True, True),
            "neither": (False, False),
        }[interval.closed]  # fmt: skip
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
        r = (x >= self.lower_bound) if self.lower_included else (x > self.lower_bound)
        return self.backend.where(self.backend.is_null(x), self.backend.true_like(x), r)

    def upper_satisfied(self, x: Arr) -> Arr:
        r"""Return a boolean mask for the upper boundary (true: value ok)."""
        if self.upper_bound is None:
            return self.backend.true_like(x)
        r = (x <= self.upper_bound) if self.upper_included else (x < self.upper_bound)
        return self.backend.where(self.backend.is_null(x), self.backend.true_like(x), r)

    def fit(self, data: Arr, /) -> None:
        # select the backend
        self.backend: Backend = get_backend(data)

        # set lower_bound
        if self.lower_bound is UNDEFINED:
            self.lower_bound = self.backend.nanmin(data)
        elif is_null_scalar(self.lower_bound):
            self.lower_bound = None

        # set upper_bound
        if self.upper_bound is UNDEFINED:
            self.upper_bound = self.backend.nanmax(data)
        elif is_null_scalar(self.upper_bound):
            self.upper_bound = None

        # set lower_value
        if self.lower_bound is None:
            self.lower_value = self.backend.to_tensor(float("-inf"))
        elif self.lower_mode is self.MODES.mask:
            self.lower_value = self.backend.to_tensor(float("nan"))
        elif self.lower_mode is self.MODES.clip:
            self.lower_value = self.lower_bound
        else:
            raise NotImplementedError

        # set upper_value
        if self.upper_bound is None:
            self.upper_value = self.backend.to_tensor(float("+inf"))
        elif self.upper_mode is self.MODES.mask:
            self.upper_value = self.backend.to_tensor(float("nan"))
        elif self.upper_mode is self.MODES.clip:
            self.upper_value = self.upper_bound
        else:
            raise NotImplementedError

    def encode(self, data: Arr, /) -> Arr:
        # NOTE: frame.where(cond, other) replaces with other if condition is false!
        data = self.backend.where(self.lower_satisfied(data), data, self.lower_value)
        data = self.backend.where(self.upper_satisfied(data), data, self.upper_value)
        return data

    def decode(self, data: Arr, /) -> Arr:
        return data
