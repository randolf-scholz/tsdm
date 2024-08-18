r"""Generic kernels that work for a variety of backends."""

__all__ = [
    # Functions
    "false_like",
    "is_scalar",
    "is_singleton",
    "true_like",
]

from math import prod

import pyarrow as pa

from tsdm.types.protocols import NumericalArray, SupportsShape


def false_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    z = x == x
    return z ^ z


def true_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    # NOTE: cannot use ~false_like(x) because for float types:
    #   `(𝙽𝚊𝙽 == 𝙽𝚊𝙽) == False and (𝙽𝚊𝙽 != 𝙽𝚊𝙽) == True`
    z = x == x
    return z ^ (~z)


def is_singleton(x: SupportsShape, /) -> bool:
    r"""Determines whether a tensor like object has a single element."""
    return prod(x.shape) == 1


def is_scalar(x: object, /) -> bool:
    r"""Determines whether an object is a scalar."""
    return isinstance(x, bool | float | int | pa.Scalar | str)
