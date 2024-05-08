r"""Generic kernels that work for a variety of backends."""

__all__ = [
    "false_like",
    "is_scalar",
    "is_singleton",
    "to_scalar",
    "true_like",
]

from math import prod

import pyarrow as pa

from tsdm.types.protocols import NumericalArray, SupportsShape


def false_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    z = x == x  # pylint: disable=comparison-with-itself
    return z ^ z


def true_like(x: NumericalArray, /) -> NumericalArray[bool]:
    r"""Returns a constant boolean tensor with the same shape/device as `x`."""
    # NOTE: cannot use ~false_like(x) because for float types:
    #   `(𝙽𝚊𝙽 == 𝙽𝚊𝙽) == False and (𝙽𝚊𝙽 != 𝙽𝚊𝙽) == True`
    z = x == x  # pylint: disable=comparison-with-itself
    return z ^ (~z)


def is_singleton(x: SupportsShape, /) -> bool:
    r"""Determines whether a tensor like object has a single element."""
    return prod(x.shape) == 1
    # numpy: size, len  / shape + prod
    # torch: size + prod / numel / shape + prod
    # table: shape + prod
    # DataFrame: shape + prod
    # Series: shape + prod
    # pyarrow table: shape + prod
    # pyarrow array: ????


def is_scalar(x: object, /) -> bool:
    r"""Determines whether an object is a scalar."""
    return isinstance(x, bool | float | int | pa.Scalar | str)


def to_scalar():
    r"""Convert a singleton to a scalar."""
    raise NotImplementedError
