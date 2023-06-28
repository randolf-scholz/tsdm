"""Universal backend functions that are independent of the backend used."""

__all__ = [
    "false_like",
    "true_like",
    "is_singleton",
]

from math import prod
from typing import TypeVar

from tsdm.types.protocols import NumericalArray, SupportsShape

A = TypeVar("A", bound=NumericalArray)
"""A type variable for numerical objects."""


def false_like(x: A, /) -> A:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    z = x == x
    return z ^ z


def true_like(x: A, /) -> A:
    """Returns a constant boolean tensor with the same shape/device as `x`."""
    # NOTE: (NAN == NAN) == False and (NAN != NAN) == True
    z = x == x
    return z ^ (~z)


def is_singleton(x: SupportsShape) -> bool:
    """Determines whether a tensor like object has a single element."""
    return prod(x.shape) == 1
    # numpy: size, len  / shape + prod
    # torch: size + prod / numel / shape + prod
    # table: shape + prod
    # DataFrame: shape + prod
    # Series: shape + prod
    # pyarrow table: shape + prod
    # pyarrow array: ????
