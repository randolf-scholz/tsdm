r"""Simple interval class."""

__all__ = ["Interval", "HalfOpenInterval", "OpenInterval", "ClosedInterval"]

from dataclasses import KW_ONLY, dataclass
from typing import Literal as L

from tsdm.pprint import pprint_repr


@pprint_repr
@dataclass(slots=True, frozen=True)
class Interval[T, Left: bool, Right: bool]:
    r"""Simple interval class."""

    left: T
    right: T
    _: KW_ONLY
    left_closed: Left
    right_closed: Right


type HalfOpenInterval[T] = Interval[T, L[True], L[False]]
type OpenInterval[T] = Interval[T, L[False], L[False]]
type ClosedInterval[T] = Interval[T, L[True], L[True]]
